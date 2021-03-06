import tensorflow as tf
import time

from .transformer_core import *


train_step_signature = [
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float64),
]


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, training, 
               look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)


        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1

    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                   maximum_position_encoding, inp_dim,
                   rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

    
    
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Input((None, None, inp_dim)),
          tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        

    def call(self, x, training, 
               look_ahead_mask, padding_mask):

        x = self.input_layer(x)

        seq_len = tf.shape(x)[1]
        attention_weights = {}


        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
    
    
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights







class Transformer(tf.keras.Model):
    
    
    def __init__(self, num_layers_enc, num_layers_dec, d_model, num_heads, dff,
                   maximum_position_encoding, net_info, inp_dim, final_dim, config,
                   rate=0.1):

        super(Transformer, self).__init__()
        
        self.decoder = Decoder(num_layers_dec, d_model, num_heads, dff, maximum_position_encoding, inp_dim,
                               rate)

        self.final_layer = tf.keras.layers.Dense(d_model, activation=None)
        

        self.ORDER = config["ORDER"]
        self.FIELD_STARTS_IN = config["FIELD_STARTS_IN"]
        self.FIELD_DIMS_IN = config["FIELD_DIMS_IN"] 
        self.FIELD_STARTS_NET = config["FIELD_STARTS_NET"]
        self.FIELD_DIMS_NET = config["FIELD_DIMS_NET"]
        self.ACTIVATIONS = config["ACTIVATIONS"]
        
        
        for name, dim in self.FIELD_DIMS_NET.items():
            acti = self.ACTIVATIONS.get(name, None)
            self.__setattr__(name, tf.keras.layers.Dense(dim, activation=acti))
        
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.results = dict([(x, []) for x in ["loss", "val_loss", "val_loss_full","parts"]])

        
        
    def call(self, tar, training,
               look_ahead_mask, dec_padding_mask):


        tar_inp = tar[:, :-1] # predict next from this
        tar_out = tar[:, 1:] 
        

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar_inp, training, look_ahead_mask, dec_padding_mask)


        final_output = self.final_layer(dec_output)
        preds = {}
        

        for net_name in self.ORDER:
            pred = self.__getattribute__(net_name)(final_output)

            preds[net_name] = pred


        return preds, attention_weights



    def train_step(self, inp, tar):


        combined_mask, dec_padding_mask = create_masks(tar)
        
        with tf.GradientTape() as tape:
            predictions, _ = self(inp, 
                                         True, 
                                         combined_mask, 
                                         dec_padding_mask)
            
            loss, *_ = self.loss_function(tar, predictions)


        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)



    def val_step(self, inp, tar):

        combined_mask, dec_padding_mask = create_masks(tar)

        predictions, _ = self(inp, 
                                     False, 
                                     combined_mask, 
                                     dec_padding_mask)
        
        return self.loss_function(tar, predictions)




    def fit(self, train_batches, x_cv, y_cv, epochs, early_stop=2, print_every=50, ckpt_every=2, mid_epoch_updates=None):
        warned_acc = False
        
        if mid_epoch_updates:
            batch_per_update = len(train_batches)// mid_epoch_updates

        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()

            for (batch_no, (inp, tar)) in enumerate(train_batches):
                self.train_step(inp, tar)
                
             
                if batch_no % print_every == 0:
                    print(f'Epoch {epoch + 1} Batch {batch_no} Loss {self.train_loss.result():.4f}')
                    
                    
                if mid_epoch_updates:
                    if batch_no % batch_per_update == 0:
                        v_loss, *vl_parts = self.val_step(x_cv, y_cv)
                        if len(vl_parts) == 1: 
                            vl_parts = vl_parts[0]
                            
                            
                        self.results["loss"].append(self.train_loss.result().numpy())
                        self.results["val_loss"].append(v_loss)
                        self.results["parts"].append(vl_parts)

                        try:
                            acc_res = self.acc_function()

                            acc_list = self.results.get("val_acc", [])
                            acc_list.append(acc_res)
                            self.results["val_acc"] = acc_list
                        except Exception as e:
                            if not warned_acc:
                                warned_acc = True
                                print("Not recording acc:", e)




            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f}')

            v_loss, *vl_parts = self.val_step(x_cv, y_cv)
            if len(vl_parts) == 1: 
                vl_parts = vl_parts[0]
                
            print(f"** on validation data loss is {v_loss:.4f}")
            
 

            self.results["loss"].append(self.train_loss.result().numpy())
            self.results["val_loss"].append(v_loss)
            self.results["parts"].append(vl_parts)
            
            try:
                acc_res = self.acc_function()
                
                acc_list = self.results.get("val_acc", [])
                acc_list.append(acc_res)
                self.results["val_acc"] = acc_list
                print(f"** on validation data acc is \n{acc_res}")
            except Exception as e:
                if not warned_acc:
                    warned_acc = True
                    print("Not recording acc:", e)
                    

            

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

            if min(self.results["val_loss"] ) < min(self.results["val_loss"][-early_stop:] ):
                print(f"Stopping early, last {early_stop} val losses are: {self.results['val_loss'][-early_stop:]} \
                      \nBest was {min(self.results['val_loss'] ):.3f}\n\n")
                break
                
                
            if (epoch + 1) % ckpt_every == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')



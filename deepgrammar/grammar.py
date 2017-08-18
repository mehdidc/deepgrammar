from grammaropt.grammar import build_grammar

grammar = r"""
archi = global_stmts archi_stmts optim_stmts

global_stmts = activation_stmt reg_stmt batchnorm_stmt
reg_stmt = "l2_coef = " l2 "\n"
l2 = "1e-4"
activation_stmt = "activation = " activation "\n"
batchnorm_stmt = "use_batch_norm = " bool "\n" 

archi_stmts = conv_archi_stmts flatten fc_archi_stmts
conv_archi_stmts = (conv_archi_stmt conv_archi_stmts) / conv_archi_stmt
conv_archi_stmt = conv batchnorm act_layer (dropout / none) (pooling / none)
fc_archi_stmts = (fc_archi_stmt fc_archi_stmt) / fc_archi_stmt
fc_archi_stmt = fc batchnorm act_layer (dropout / none)
fc = "x = Dense(" nb_units ", activation='linear', kernel_initializer=" init ", kernel_regularizer=l2(l2_coef))(x)\n"
batchnorm = "x = BatchNormalization(axis=batch_norm_axis)(x) if use_batch_norm else x\n"
conv = "x = " conv_layer "(" filters ", " kernel_size ", strides=" strides ", activation='linear', kernel_initializer=" init ", kernel_regularizer=l2(l2_coef), padding=" padding ")(x)\n"
conv_layer = "Conv2D"
act_layer = "x = Activation(activation)(x)\n"

flatten = "x = Flatten()(x)\n"
padding = "\"valid\"" / "\"same\""
filters = "16" / "32" / "48" / "64" / "80" / "96" / "112"
kernel_size =  "1" / "3" / "5" / "7"
strides = "(1, 1)" / "(2, 2)"
activation = "\"relu\"" / "\"elu\"" 
pooling = "x = " poolingop "(2)(x)\n"
poolingop = "MaxPooling2D"
init = "\"glorot_uniform\""
dropout = "x = Dropout(" proba ")(x)\n"
proba = "0.1" / "0.3"/ "0.5"

optim_stmts = lr_stmt "\n" epochs_stmt "\n" batch_size_stmt "\n" optimizer_stmt "\n"
optimizer_stmt = "opt = optimizers." opt
epochs_stmt = "epochs = " epochs 
batch_size_stmt = "batch_size = " batch_size
lr_stmt = "lr = " lr
opt = sgd
decay = "0.0"
sgd = "SGD(lr=lr, momentum=" momentum ", decay=" decay ", nesterov=" nesterov ")"
lr = "0.1" / "0.05" / "0.01" / "0.005" / "0.001"
batch_size = "64"
momentum = "0.9"
nesterov = "True"
bool = "True" / "False"
epochs = "100" 
nb_units = "128" / "256" / "512" / "1024" / "2048"
none = ""
"""
grammar = build_grammar(grammar)

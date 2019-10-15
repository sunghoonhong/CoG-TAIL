#system info
DEVICE = 'cuda:0'
EXPERT_DIR = './restricted_expert_data'
EXPERT_CHUNKS_NUM = 400
EXPERT_CHUNK_LENGTH = 512
BAD_TOKENS = [1, 31, 2, 15, 11, 251]
#<SEP> ? ! . * 10

#for bert encoder
PRETRAINED_WEIGHTS = 'bert-base-uncased'

#for environment
STATE_SIZE = 768
CODE_SIZE = 2
CLS_TOKEN_IDX = 0
SEP_TOKEN_IDX = 1
GEN_MAX_LEN = 32
TRAIN_REPORT_PERIOD = 50

#batch size
BATCH_SIZE = 64

#for discriminator
DISC_HIDDEN_UNIT_NUM = STATE_SIZE
DISC_LATENT_SIZE = 8
DISC_LR = 1e-4
WEIGHT_FOR_CODE = 0.5
WEIGHT_FOR_KL = 1e-3
KL_STEP = 1e-5
IC = 0.5
DISC_STEP = 2
DISC_UPDATE_CNT = 1

#for actor_critic
AC_HIDDEN_UNIT_NUM = 2000
AC_HIDDEN_UNIT_STRIDE = 500
AC_HIDDEN_LAYER_NUM = 2
AC_LR = 5e-5
PRETRAIN_LR = 5e-5
CRITIC_HIDDEN_UNIT_NUM = 500
TOP_K = 40
EPSILON = 0.15
ENTROPY = 1e-1
ACTOR_COEF = 1
CRITIC_COEF = 1
PRETRAIN_COEF = 0.5
PPO_STEP = 5
PRETRAIN_SAVEPATH = './model_save/pretrain.pt'

#for memory
GAMMA = 0.98
LAMBDA = 0.95
THRESHOLD_LEN = 512
HORIZON_THRESHOLD = 35

#for statistics
MOVING_AVERAGE = 50

#for action_autoencoder
VOCAB_SIZE = 5000
AUTOENCODER_HIDDEN_UNIT_NUM = 300
COMPRESSED_VOCAB_SIZE = 8
AUTOENCODER_BATCH_SIZE = 128
AUTOENCODER_KL_COEF = 0.1
AUTOENCODER_SAVE_PATH = './model_save/encoder'
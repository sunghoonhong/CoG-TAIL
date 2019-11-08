#system info
DEVICE = 'cuda:0'
EXPERT_DIR = './batch_cls'
EXPERT_CHUNKS_NUM = 200
EXPERT_CHUNK_LENGTH = 512
BAD_TOKENS = []

#for bert encoder
PRETRAINED_WEIGHTS = 'bert-base-uncased'

#for environment
STATE_SIZE = 768
CODE_SIZE = 2
CLS_TOKEN_IDX = 1
SEP_TOKEN_IDX = 0
GEN_MAX_LEN = 12
TRAIN_REPORT_PERIOD = 50
MODEL_SAVE_PERIOD = 10000
ENCODING_FLAG = 'FIRST'

#batch size
BATCH_SIZE = 128

#for discriminator
DISC_HIDDEN_UNIT_NUM = STATE_SIZE
DISC_LATENT_SIZE = 32
DISC_LR = 1e-4
WEIGHT_FOR_CODE = 1
WEIGHT_FOR_KL = 1e-1
KL_STEP = 1e-3
IC = 0.3
DISC_STEP = 2
DISC_UPDATE_CNT = 1
AFTER_TRAIN_DISC_UPDATE_CNT = 1

#for actor_critic
AC_HIDDEN_UNIT_NUM = 512
AC_HIDDEN_UNIT_STRIDE = 500
AC_HIDDEN_LAYER_NUM = 1
AC_LR = 1e-4
PRETRAIN_LR = 3e-4
CRITIC_HIDDEN_UNIT_NUM = 32
TOP_K = 40
EPSILON = 0.2
ENTROPY = 1e-2
ACTOR_COEF = 1
CRITIC_COEF = 0.1
PRETRAIN_COEF = 5e-2
INFO_COEF = 1
PPO_STEP = 5
PRETRAIN_SAVEPATH = './model_save/pretrain.pt'
MODEL_SAVEPATH = './model_save/trained'

#for memory
GAMMA = 0.99
LAMBDA = 0.95
THRESHOLD_LEN = 512
HORIZON_THRESHOLD = 15

#for statistics
MOVING_AVERAGE = 50

#for action_autoencoder
VOCAB_SIZE = 3600
AUTOENCODER_HIDDEN_UNIT_NUM = 300
COMPRESSED_VOCAB_SIZE = 8
AUTOENCODER_BATCH_SIZE = 128
AUTOENCODER_KL_COEF = 0.1
AUTOENCODER_SAVE_PATH = './model_save/encoder'

#for action_encoder
ACTIONENCODER_BATCH_SIZE = 128
ACTIONENCODER_SAVE_PATH = './model_save/actionencoder.pt'

#tmp
REWARD_LIST = []
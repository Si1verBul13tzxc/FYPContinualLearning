import argparse


def asc_config(parser):
    parser.add_argument('--backbone', default='', type=str, required=True,
                        help='chose the backbone model',
                        choices=['bert', 'bert_frozen', 'bert_adapter', 'w2v_as', 'w2v', 'cnn', 'mlp'])

    parser.add_argument('--baseline', default='', type=str, required=True,
                        help='chose the baseline model',
                        choices=['ncl', 'one', 'mtl', 'l2', 'a-gem', 'derpp', 'kan', 'srk', 'ewc', 'hal', 'ucl', 'owm',
                                 'acl', 'hat', 'cat', 'b-cl', 'classic', 'ctr'])
    parser.add_argument('--task', default='', type=str, required=True, help='what datasets',
                        choices=['asc', 'dsc', 'ssc', 'nli', 'newsgroup', 'celeba', 'femnist', 'vlcs', 'cifar10',
                                 'mnist', 'fashionmnist', 'cifar100'])
    parser.add_argument('--scenario', default='', type=str, required=True, help='what senario it will be',
                        choices=['til_classification', 'dil_classification']
                        )

    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--note', type=str, default='', help='(default=%(default)d)')
    parser.add_argument('--nclasses', default=0, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--ntasks', default=10, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--ntasks_unseen', default=10, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--idrandom', default=0, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr_patience', default=5, type=int, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_factor', default=3, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_gap', default=0.01, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_min', default=1e-4, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--thres_cosh', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--thres_emb', default=6, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--clipgrad', default=10000, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lamb', default=5000, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--sbatch', default=64, type=int, required=False, help='(default=%(default)f)')
    parser.add_argument('--output_dir', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--model_path', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--bert_mask_adapter_size', default=2000, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--bert_adapter_size', default=2000, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--mlp_adapter_size', default=2000, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--semantic_cap_size', default=3, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--num_semantic_cap', default=3, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--mid_size', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--experiment_id', type=int, default=0)
    parser.add_argument('--use_predefine_args', action='store_true')
    parser.add_argument('--temp', type=float, default=1,
                        help='temperature for loss function')
    parser.add_argument('--base_temp', type=float, default=1,
                        help='temperature for loss function')
    parser.add_argument('--buffer_percent', type=float, required=False,
                        help='The size of the memory buffer.')
    parser.add_argument('--buffer_size', type=int, required=False,
                        help='The size of the memory buffer.')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--sup_loss', action='store_true')
    parser.add_argument('--distill_loss', action='store_true')
    parser.add_argument('--my_contrast', action='store_true')
    parser.add_argument('--trans_loss', action='store_true')
    parser.add_argument('--true_id', action='store_true')
    parser.add_argument('--larger_as_list', action='store_true')
    parser.add_argument('--resume_model', action='store_true')
    parser.add_argument('--head_ewc', action='store_true')
    parser.add_argument('--head_robust', action='store_true')
    parser.add_argument('--train_twice', action='store_true')
    parser.add_argument('--augment_current', action='store_true')
    parser.add_argument('--augment_distill', action='store_true')
    parser.add_argument('--augment_trans', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpus', type=int, default=0)
    parser.add_argument('--resume_from_task', type=int, default=0)
    parser.add_argument('--resume_from_file', type=str, default='')
    parser.add_argument('--sample_gate_in_ouput', action='store_true')
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--use_pooled_rep', action='store_true')
    parser.add_argument('--pooled_rep_contrast', action='store_true')
    parser.add_argument('--task_gate_in_ouput', action='store_true')
    parser.add_argument('--mtl', action='store_true')
    parser.add_argument('--overlap_only', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_each_step', action='store_true')
    parser.add_argument('--eval_each_step', action='store_true')
    parser.add_argument('--known_id', action='store_true')
    parser.add_argument('--first_id', action='store_true')
    parser.add_argument('--last_id', action='store_true',
                        help='use last ID as the ID for testing, useful in DIL setting')
    parser.add_argument('--ent_id', action='store_true', help='use entropy to decide ID, useful in DIL setting')
    parser.add_argument('--entropy_loss', action='store_true')
    parser.add_argument('--share_conv', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--build_adapter', action='store_true')
    parser.add_argument('--build_adapter_ucl', action='store_true')
    parser.add_argument('--build_adapter_owm', action='store_true')
    parser.add_argument('--build_adapter_mask', action='store_true')
    parser.add_argument('--build_adapter_capsule_mask', action='store_true')
    parser.add_argument('--build_adapter_capsule', action='store_true')
    parser.add_argument('--apply_bert_output', action='store_true')
    parser.add_argument('--apply_bert_attention_output', action='store_true')
    parser.add_argument('--apply_one_layer_shared', action='store_true')
    parser.add_argument('--apply_two_layer_shared', action='store_true')
    parser.add_argument('--transfer_layer_incremental', action='store_true')
    parser.add_argument('--transfer_layer_all', action='store_true')
    parser.add_argument('--task_mask', action='store_true')
    parser.add_argument('--no_tsv_mask', action='store_true')
    parser.add_argument('--tsv_mask_type', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gradient_accumulation', type=float, default=2)
    parser.add_argument('--eval_steps', type=float, default=200)
    parser.add_argument('--logging_steps', type=float, default=200)
    parser.add_argument('--xusemeval_num_train_epochs', type=int, default=0)
    parser.add_argument('--bingdomains_num_train_epochs', type=int, default=0)
    parser.add_argument('--bingdomains_num_train_epochs_multiplier', default=0)
    parser.add_argument('--w2v_hidden_size', type=int, default=300)
    parser.add_argument('--capsule_nhid', type=int, default=2000)
    parser.add_argument('--capsule_nhid_output', type=int, default=768)
    parser.add_argument('--cut_partition', type=float, default=1)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--model_checkpoint', type=str, default='')
    parser.add_argument('--l2_norm', action='store_true')
    parser.add_argument('--larger_as_share', action='store_true')
    parser.add_argument('--mask_head', action='store_true')
    parser.add_argument('--sup_head', action='store_true')
    parser.add_argument('--distill_head', action='store_true')
    parser.add_argument('--amix_head', action='store_true')
    parser.add_argument('--current_head', action='store_true')
    parser.add_argument('--transfer_route', action='store_true')
    parser.add_argument('--no_capsule', action='store_true')
    parser.add_argument('--momentum', action='store_true')
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--srk_train_batch_size', type=int, default=32)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--class_per_task', type=int, default=2)
    parser.add_argument('--use_imp', action='store_true')
    parser.add_argument('--use_gelu', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--data_size', type=str)
    parser.add_argument('--image_size', type=int, default=0)
    parser.add_argument('--image_channel', type=int, default=0)
    parser.add_argument('--train_data_size', type=int, default=0)
    parser.add_argument('--train_data_size_ontonote', type=int, default=0)
    parser.add_argument('--dev_data_size', type=int, default=0)
    parser.add_argument('--test_data_size', type=int, default=0)
    parser.add_argument('--cnn_kernel_size', type=int, default=100)
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--max_pred', type=int, default=128,
                        help="Max tokens of prediction.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warm_train", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    return parser


def train_config(parser):
    ## Other parameters

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--bert_hidden_size", default=768, type=int)
    parser.add_argument("--bert_num_hidden_layers", default=12, type=str)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_term_length",
                        default=5,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_sentence_length",
                        default=123,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument('--data_seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    # attention
    parser.add_argument('--deep_att_lexicon_input_on', action='store_false')
    parser.add_argument('--deep_att_hidden_size', type=int, default=64)
    parser.add_argument('--deep_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--deep_att_activation', type=str, default='relu')
    parser.add_argument('--deep_att_norm_on', action='store_false')
    parser.add_argument('--deep_att_proj_on', action='store_true')
    parser.add_argument('--deep_att_residual_on', action='store_true')
    parser.add_argument('--deep_att_share', action='store_false')
    parser.add_argument('--deep_att_opt', type=int, default=0)

    # self attn
    parser.add_argument('--self_attention_on', action='store_false')
    parser.add_argument('--self_att_hidden_size', type=int, default=64)
    parser.add_argument('--self_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--self_att_activation', type=str, default='relu')
    parser.add_argument('--self_att_norm_on', action='store_true')
    parser.add_argument('--self_att_proj_on', action='store_true')
    parser.add_argument('--self_att_residual_on', action='store_true')
    parser.add_argument('--self_att_dropout', type=float, default=0.1)
    parser.add_argument('--self_att_drop_diagonal', action='store_false')
    parser.add_argument('--self_att_share', action='store_false')
    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.4)
    # query summary
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('--down_sample_ratio', type=float, default=1)

    return parser


def load_pre_defined_args(args):
    # ============= dataset base ==================
    if args.task == 'asc':  # aspect sentiment classification
        args.ntasks = 10
        args.num_train_epochs = 10
        args.xusemeval_num_train_epochs = 10
        args.bingdomains_num_train_epochs = 30
        args.bingdomains_num_train_epochs_multiplier = 3
        args.nepochs = 100
        args.nclasses = 3

    # ============= backbone base ==================
    if 'bert_adapter' in args.backbone:
        args.apply_bert_output = True
        args.apply_bert_attention_output = True

    # ============= approach base ==================
    if args.baseline == 'ctr':
        args.apply_bert_output = True
        args.apply_bert_attention_output = True
        args.build_adapter_capsule_mask = True
        args.apply_one_layer_shared = True
        args.use_imp = True
        args.transfer_route = True
        args.share_conv = True
        args.larger_as_share = True
        args.adapter_size = True

    if args.baseline == 'b-cl':
        args.apply_bert_output = True
        args.apply_bert_attention_output = True
        args.build_adapter_capsule_mask = True
        args.apply_one_layer_shared = True
        args.semantic_cap_size = 3

    return args

def set_args():
    parser = argparse.ArgumentParser()
    parser = asc_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    if args.use_predefine_args:
        args = load_pre_defined_args(args)
    # print(args)
    return args
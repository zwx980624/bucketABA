import argparse



parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--max_basket_num', type=int, default=10, help='max_basket_num')
parser.add_argument('--max_basket_size', type=int, default=10, help='max_basket_num')
parser.add_argument('--cvr_neg_ratio', type=int, default=1, help='cvr_neg_ratio')
parser.add_argument('--CVR_sample_rate', type=float, default=1.0, help='CVR_sample_rate')
parser.add_argument('--CVABR_balance', type=float, default=1.0, help='CVABR_balance')
parser.add_argument('--CVBCVAR_balance', type=float, default=1.0, help='CVBCVAR_balance')
parser.add_argument('--data_path', type=str, default="./my_data/Dunn/user_date_tran_dict_new.txt", help='data_path')
parser.add_argument('--only_test', action="store_true")



args = parser.parse_args()
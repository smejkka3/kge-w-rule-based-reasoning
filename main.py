from data.load.load_data import LoadData
from reasoner.reasoner import Reasoner
from generator.generator import Generator
from evaluator.evaluate import Evaluate
import pandas as pd
import os
import subprocess
import sys
from loguru import logger

logger.add("logger.log", rotation="500 MB")

def main():
    file_names = [  '/home/smejkal/github/kge-w-rule-based-reasoning/data/yago/yago-wd-schema.nt', 
                    '/home/smejkal/github/kge-w-rule-based-reasoning/data/yago/yago-wd-full-types.nt',
                    '/home/smejkal/github/kge-w-rule-based-reasoning/data/yago/yago-wd-facts.nt']

    dfs = []

    for file in file_names:
        dataLoader = LoadData(file)
        if file.endswith('schema.nt'):
            df = dataLoader.load_data(check_schema=True)
        else:
            df = dataLoader.load_data(testing=True)
        dfs.append(df)

    train_df = pd.concat(dfs, ignore_index=True).assign(prob=1.0)
    
    while True:
        reasoner = Reasoner(train_df)
        reasoner.apply_rules()
        reasoner.split_and_save_for_training()

        logger.info(f"Processing data...")
        subprocess.run([sys.executable, "data/preprocess/preprocess_default.py", "data/train_data/"])
        logger.info(f"Training...")
        os.system('kge start --console.quiet True model/transe-train.yaml --job.device cpu')
        logger.info(f"Training complete...")

        generator = Generator("/home/smejkal/github/kge-w-rule-based-reasoning/data/train_data/train.del")
        candidates = generator.generate_new_triples()

        evaluator = Evaluate(candidates)
        new_triples_df = evaluator.evaluate_and_return_facts()

        if not new_triples_df.empty:
            new_triples_df.rename(columns = {3:'prob'}, inplace = True)
            df_list = [train_df,new_triples_df]
            train_df  = pd.concat(df_list, ignore_index=True)
            train_df.loc[:,[0,1,2]].to_csv('/home/smejkal/github/kge-w-rule-based-reasoning/data/train_data/train.txt',sep='\t', index=False, header=False)
        else:
            logger.info(f"Program finished, no new triples found.")
            break

if __name__ == "__main__":
    main()
#print(new_df.info())





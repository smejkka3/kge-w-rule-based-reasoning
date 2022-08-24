import pandas as pd
from loguru import logger

class LoadData:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self, check_schema=False, testing=False):
        logger.info(f"Reading data {self.filename} ..")
        self.df = pd.read_csv(self.filename, sep='\t',header=None,
                         names=None,
                         dtype=str,
                         usecols=[0, 1, 2])
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna().reset_index(drop=True)
        for column in self.df.columns:
            self.df[column] = self.df[column].str.replace(",",";",regex=True) 
        
        if check_schema == True:
            self.df = self.df[self.df[1].str.contains('type|subClassOf|subPropertyOf|range|domain')]
        
        if testing == True:
            #samples = int(input("Enter number of samples\n"))
            self.df = self.df.sample(n=50000, random_state=1).reset_index(drop=True)
        logger.info(f"{self.filename} was loaded succesfully...")
        return self.df
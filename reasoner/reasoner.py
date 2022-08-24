import problog
import pandas as pd

from sklearn.model_selection import train_test_split
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from problog.logic import Var, Term, Constant
from problog.program import SimpleProgram
from loguru import logger

class Reasoner:
    def __init__(self, df):
        self.triples = df.values.tolist()
    
    def apply_rules(self):
        logger.info(f"applying reasoner...")
        TRIPLE = Term('triple')
        query = Term('query')
        SUBCLASS = Term('<http://www.w3.org/2000/01/rdf-schema#subClassOf>')
        TYPE_ = Term('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')
        DOMAIN_ = Term('<http://www.w3.org/2000/01/rdf-schema#domain>')
        RANGE_ = Term('<http://www.w3.org/2000/01/rdf-schema#range>')
        SUBPROPERTY = Term('<http://www.w3.org/2000/01/rdf-schema#subPropertyOf>')

        C3, C2, C1, X, Y, R1, R2, R3 = Var('C3'), Var('C2'), Var('C1'), Var('X'), Var('Y'), \
                                                Var('R1'), Var('R2'), Var('R3')
        rdfs2 = TRIPLE(X, TYPE_, C1) << ( TRIPLE(R1, DOMAIN_, C1) & TRIPLE(X, R1, Y) )
        rdfs3 = TRIPLE(Y, TYPE_, C1) << ( TRIPLE(R1, RANGE_, C1) & TRIPLE(X, R1, Y) )
        rdfs5 = TRIPLE(R1, SUBPROPERTY, R3) << ( TRIPLE(R1, SUBPROPERTY, R2) & TRIPLE(R2, SUBPROPERTY, R2) )
        rdfs7 = TRIPLE(X, R2, Y) << ( TRIPLE(R1, SUBPROPERTY, R2) & TRIPLE(X, R1, Y) )
        rdfs9 = TRIPLE(X, TYPE_, C2) << ( TRIPLE(C1, SUBCLASS, C2) & TRIPLE(X, TYPE_, C1) )
        rdfs11 = TRIPLE(C1, SUBCLASS, C3) << ( TRIPLE(C1, SUBCLASS, C2) & TRIPLE(C2, SUBCLASS, C3) )

        p = SimpleProgram()

        for t in self.triples:
            p += TRIPLE(Term(t[0]),Term(t[1]),Term(t[2]), p=t[3]) 
            
        p += rdfs2
        p += rdfs3
        p += rdfs5
        p += rdfs7
        p += rdfs9
        p += rdfs11

        p += query(TRIPLE(X, TYPE_, C1))
        p += query(TRIPLE(Y, TYPE_, C1))
        p += query(TRIPLE(R1, SUBPROPERTY, R3))
        p += query(TRIPLE(X, R2, Y))
        p += query(TRIPLE(X, TYPE_, C2))
        p += query(TRIPLE(C1, SUBCLASS, C3))

        result = get_evaluatable().create_from(p).evaluate()
        result = {key:val for key, val in result.items() if val > 0.8}

        lst = []
        for triple in result:
            try:
                lst.append((str(triple)[7:-1].split(',')[0], str(triple).split(',')[1], str(triple)[:-1].split(',')[2]))
            except:
                logger.info(f"Error adding: {triple}")
                pass
        self.df = pd.DataFrame(lst, columns =['S','P','O']).assign(prob=result.values())

    
    def split_and_save_for_training(self):
        logger.info(f"splitting data...")
        X_train, X_test = train_test_split(self.df, test_size=0.05, random_state=1)
        X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=1)

        X_train.to_csv('/home/smejkal/github/kge-w-rule-based-reasoning/data/train_data/train.txt',sep='\t', index=False, header=False)
        X_test.to_csv('/home/smejkal/github/kge-w-rule-based-reasoning/data/train_data/test.txt',sep='\t', index=False, header=False)
        X_val.to_csv('/home/smejkal/github/kge-w-rule-based-reasoning/data/train_data/valid.txt',sep='\t', index=False, header=False)


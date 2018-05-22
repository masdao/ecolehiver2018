import torch
import numpy as numpy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1234)

titanic_df = pd.read_csv('https://github.com/afansi/winterschool18/blob/master/titanic3.csv?raw=true',
    sep='\t', 
    index_col=None, 
    na_values=['NA']
)
titanic_df.head()

titanic_preprocess_df = pd.read_csv(
    '/Users/ms/dev/ecolehiver2018/data/titanic_prepocess.csv', 
    sep=',', 
    index_col=None
)

titanic_preprocess_df.head()

np.random.seed(1234)
train, validate, test = np.split(
    titanic_preprocess_df.sample(frac=1, random_state=134), 
    [int(.6*len(titanic_preprocess_df)), 
     int(.8*len(titanic_preprocess_df))
    ]
)

X_train = train.drop(['survived'], axis=1).values
y_train = train['survived'].values

X_val = validate.drop(['survived'], axis=1).values
y_val = validate['survived'].values

X_test = test.drop(['survived'], axis=1).values
y_test = test['survived'].values

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()        
        self.fc1 = nn.Linear(12, 20)
        
        # À Completer avec la définition d'autres couches .....
        

    def forward(self, x):
       x = F.relu(self.fc1(x))
    
       # À Completer avec l'appel à d'autres couches .....
    
       
       return x


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import wrangle as w



def plot(train,col):
    
    crosstab = pd.crosstab(train[col], train.y_encoded)
    
    per_crosstab = crosstab.div(crosstab.sum(axis=1),axis = 0)
    
    per_crosstab.plot(kind='bar',stacked=True, figsize=(10,6), color=['#B0B0B0', '#2E8B57'])
    plt.ylim(0, 1.10)
    plt.gca().yaxis.set_major_formatter(lambda x, _: f'{x:.0%}')
    plt.xticks(rotation = 45, ha= 'right')             
    plt.xlabel(col)
    plt.ylabel('Percent')
    plt.title(f'Conversion Rate By {col}')
    plt.legend(title = 'Term Deposit', labels = ['No','Yes'], bbox_to_anchor=(1.02, 1), loc='upper left') 
    plt.show()


def chi_test(train, col):
    
    # set alpha value to 0.05
    alpha = 0.05
    
    # set null and alternative hypothesis 
    null_hypothesis = col + ' and term deposits are independent'
    alternative_hypothesis = col + ' and term deposits are dependent'

    # create an observed crosstab, or contingency table from a dataframe's two columns
    observed = pd.crosstab(train[col], train.y_encoded)

    # run chi-square test
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    # print Null Hypothesis followed by a new line
    print(f'Null Hypothesis: {null_hypothesis}\n')

    # print Alternative Hypothesis followed by a new line
    print(f'Alternative Hypothesis: {alternative_hypothesis}\n')

    # print the chi2 value
    print(f'chi^2 = {chi2}') 

    # print the p-value followed by a new line
    print(f'p     = {p}\n')

    if p < alpha:
        print(f'We reject null hypothesis')
        print(f'There exists some relationship between {col} and term deposits.')
    else:
        print(f'We fail to reject null hypothesis')
        print(f'There appears to be no significant relationship between {col} and term deposits.')
        
def log_test(train, col):
    alpha = 0.05
    H_0 = 'Age has no effect on subscription probability'
    H_a = 'Age affects subscription probability'
    
    X = sm.add_constant(train[col])
    y = train['y_encoded']
    
    model = sm.Logit(y, X).fit()
    
    coef = model.params[col]
    p_value = model.pvalues[col]
    
    print(f"Coefficient: {coef}")
    print(f"P-value: {p_value}")
    
    if p_value < alpha:
        print("Reject the null hypothesis that", H_0)
        print("Sufficient evidence that", H_a)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    
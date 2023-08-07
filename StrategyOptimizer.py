import pandas as pd
import numpy as np
from tqdm import tqdm
from Metrics import *
from plotly.subplots import make_subplots
from Strategies import *
from StrategyTester import StrategyTester
import matplotlib.pyplot as plt 
import inspect
import itertools
from apply import apply 


class StrategyOptimizer():
    #initial arguments
    
    def __init__(self, folds, test_size, symbol, data_tester,  strategy_func):
        '''
            class StrategyOptimizer for find best parameters for given strategy
            
            Args:
            =================================================
            
            folds -  int , number of splits to test strategy,
            test_size - float, part of data that will be using to forward test in given fold
            symbol - string - instrument
            data_tester - pandas df data
            strategy_func - strategy to test
        '''
        
        
        self.folds=folds
        self.test_size=test_size
        self.strategy_func=strategy_func
        self.data=data_tester
        self.symbol=symbol
        self.tester=StrategyTester(data=self.data, symbol=self.symbol, strategy_func=self.strategy_func)
        self.buy_and_hold=pd.Series()
    
    
    
    
    def __repr__(self):
        return f"StrategyOptimizer(symbol = {self.symbol}, folds={self.folds},  strategy_func={self.strategy_func})"
    
    #method to split data into sets to backtest and forward test
    def split_folds(self, data, fold, test_size):
        fold_train=[]
        fold_test=[]
        
        #test szie should be greater than 0
        if test_size==0:

            print('To forward test your strategy test size should be gretaer than 0')
        
        
        else:
           
            if fold<=1:
                train_idx=int(len(data)*(1-test_size))
                temp_fold_train=data.iloc[:train_idx].index
                temp_fold_test=data.iloc[train_idx:].index
                fold_train.append(temp_fold_train)
                fold_test.append(temp_fold_test)
                


            else:
                fold_all=None
                for i in range(fold):

                    fold_size=int(len(data)*(1-test_size))

                    idx_start=int((len(data)*test_size)/(fold-1))*i

                    if i==fold-1:
                        fold_all=data.iloc[idx_start:]
                        train_fold_end=int(fold_size*(1-test_size))

                        fold_train_temp=fold_all.iloc[:train_fold_end].index


                        fold_test_temp=fold_all.iloc[train_fold_end:].index
                        fold_train.append(fold_train_temp)
                        fold_test.append(fold_test_temp)

                    else:   

                        fold_all=data.iloc[idx_start:idx_start+fold_size]
                        train_fold_end=int(fold_size*(1-test_size))

                        fold_train_temp=fold_all.iloc[:train_fold_end].index
                        test_fold_end=train_fold_end+int(fold_size*test_size)

                        fold_test_temp=fold_all.iloc[train_fold_end:train_fold_end+int(fold_size*test_size)].index

                        fold_train.append(fold_train_temp)
                        fold_test.append(fold_test_temp)






            return fold_train, fold_test
        
    
    #Strategy optimization
    def test_params(self, **kwargs): 
        
        
        
        folds=self.folds
        check_attr_error=False
        
        #verification of argument passed to optimizer
        for i in range(len(self.tester.func_args)):
            
            if self.tester.func_args[i] not in kwargs.keys() :
                
                print(f'Define correct param for {self.strategy_func.__name__} strategy: {self.tester.func_args[i]}')
                check_attr_error=True
                
                
            
        
        if folds==0:
            raise ValueError('Folds number should be greater than 0')
        
        #splitting data into dstasets according to defined folds and test_size
        
        self.fold_train, self.fold_test=self.split_folds(self.data, self.folds, self.test_size)
        self.plot_folds(self.data, self.folds, self.test_size)
        if check_attr_error==False:
            
            
            attr_dict={}
            
            #create params for given ranges or list of parameters
            for attr in kwargs.keys():
                
                if type(kwargs[attr]) == list:
                    attr_dict[attr]=kwargs[attr]
                    
                
                else:
                    attr_dict[attr]=range(*kwargs[attr])
          
                    
            
                
            
            product = [x for x in apply(itertools.product, attr_dict.values())]
            combinations=[dict(zip(attr_dict.keys(), p)) for p in product]
            
            test_performance=[]
            performance = []
            count_help=0
            
            #iterate over possible combinations of parameters
            for comb in tqdm(combinations):
            
                
                temp_perform=[]
                perform_mean_dict={}
                
                if self.buy_and_hold.empty:
                    self.buy_and_hold=self.tester.test_strategy(**comb).iloc[:,0]
                
                
                for j in range(folds):
                    
                    
                    train_start=self.fold_train[j][0]
                    
                    train_end=self.fold_train[j][-1]
                   
                    
                    #slicing data according to range of splits
                    data_perform=(self.data.loc[train_start:train_end])
                   
                    #test strategy
                    tester_perform=StrategyTester(data=data_perform, symbol=self.symbol, strategy_func=self.strategy_func)
                    
                    #get from result dataframe only column with strategy
                    temp_perform_dict=tester_perform.test_strategy(**comb).iloc[:,1].to_dict()
                    
                    
                    temp_perform.append(temp_perform_dict)
                
               
                for k in temp_perform[0].keys():
                    #calculate mean of metrics from folds
                    perform_mean_dict[k]=sum(d[k] for d in temp_perform) / len(temp_perform)
                
                
                #create dict wit parameters and metrics
                perform_mean={**comb,**perform_mean_dict}
                
                
                
                performance.append(perform_mean)
              
              
                #dataframe with results
                self.results_overview =  pd.DataFrame(performance)
           
            
    def forward_test(self):
        func_names=['simple_return', 'mean_return', 'stddev','sharpe_ratio','sortino_ratio', 'max_dd',
                   'cagr','calmar_ratio','kelly']
        
        fold_test=self.fold_test
        folds=self.folds
        
        test_perform=[]
        results_df=self.results_overview.copy()
        
        
        for i in tqdm(range(len(results_df))):
            params=results_df.iloc[i].drop(func_names).astype(int).to_dict()
            
            temp_test_perform=[]
            test_mean_dict={}
            for j in range(folds):
                test_start=fold_test[j][0]
                test_end=fold_test[j][-1]
                data_test=(self.data.loc[test_start:test_end])
                tester_test=StrategyTester(data=data_test, symbol=self.symbol, strategy_func=self.strategy_func)
                temp_test_perform_dict=tester_test.test_strategy(**params).iloc[:,1].to_dict()
                temp_test_perform.append(temp_test_perform_dict)
            
            for k in temp_test_perform[0].keys():
                    test_mean_dict[k]=sum(d[k] for d in temp_test_perform) / len(temp_test_perform)
            
            test_mean={**params,**test_mean_dict}
            test_perform.append(test_mean)
        
        self.forward_results_overview=pd.DataFrame(test_perform)
    
    def find_best_strategies(self,metric, sort_arg='train_metric'):
        
        #metric to compare from buy_and_hold strategy
        benchmark=self.buy_and_hold[metric]
       
        
        #merge dataframes
        all_df=pd.merge(self.results_overview, self.forward_results_overview, on=self.tester.func_args,
                            suffixes=('_train', '_test'))
        
        
        df_better_than_benchmark=None
        #filter dataframe from records better than  benchmark metric
        if (metric=='max_dd') | (metric=='stddev'):
            df_better_than_benchmark=all_df.loc[((all_df[f'{metric}_train']<benchmark) & (all_df[f'{metric}_test']<benchmark))]
        else:          
            df_better_than_benchmark=all_df.loc[((all_df[f'{metric}_train']>benchmark) & (all_df[f'{metric}_test']<benchmark))]
        
        df_better_than_benchmark['distance']=abs(df_better_than_benchmark[f'{metric}_train']-df_better_than_benchmark[f'{metric}_test'])
        
        
        #new order for columns, columns with tested metrics should be first
        order_first_cols=[]
        if sort_arg=='train_metric':
            sort_arg=f'{metric}_train'
            order_first_cols=[f'{metric}_train',f'{metric}_test','distance']
            temp_new_order=[i for i in df_better_than_benchmark if i not in order_first_cols]
            order_first_cols.extend(temp_new_order)
            
            
        elif sort_arg=='test_metric':
            sort_arg=f'{metric}_test'
            order_first_cols=[f'{metric}_test',f'{metric}_train','distance']
            temp_new_order=[i for i in df_better_than_benchmark if i not in order_first_cols]
            order_first_cols.extend(temp_new_order)
            
        else:
            order_first_cols=['distance',f'{metric}_train',f'{metric}_test']
            temp_new_order=[i for i in df_better_than_benchmark if i not in order_first_cols]
            order_first_cols.extend(temp_new_order)
            
        
        
        
        
        if (metric=='max_dd') | (metric=='stddev') | (sort_arg=='distance'):
            self.best_strategies=df_better_than_benchmark.loc[:,order_first_cols].sort_values(by=sort_arg)
        
        else:
            self.best_strategies=df_better_than_benchmark.loc[:,order_first_cols].sort_values(by=sort_arg, ascending=False)
        
        return self.best_strategies

    
    def plot_folds(self, data, folds, test_size=None):
    
        data=data.reset_index()
        empty_bar_down=[]
        fold_train_data=[]
        fold_test_data=[]
        empty_bar_up=[]

        fold_train, fold_test=None, None


        if folds<=1:
            fold_train, fold_test=self.split_folds(data, folds, test_size)
            
                
            
            empty_bar_down=0
            fold_train_data=len(fold_train[0])
            fold_test_data=len(fold_test[0])
            empty_bar_up=0




        else:
            fold_train, fold_test=self.split_folds(data, folds, test_size)



            for i in range(folds):

                empty_bar_down.append(fold_train[i][0])
                fold_train_data.append(fold_train[i][-1]-fold_train[i][0])
                fold_test_data.append(fold_test[i][-1]-fold_test[i][0])
                empty_bar_up.append(len(data)-fold_test[i][-1])



        fold_plot_data=pd.DataFrame({'empty_bar_down':empty_bar_down,
                                 'train_data':fold_train_data,
                                'test_data':fold_test_data,
                                'empty_bar_up':empty_bar_up
                                    }, index=range(1,folds+1))
        fold_plot_data.plot(kind='barh', stacked=True, color=['white', 'green', 'red','white'])
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,2]
        plt.legend([handles[idx] for idx in order[:2]],[labels[idx] for idx in order[:2]])
        return fold_plot_data 
    
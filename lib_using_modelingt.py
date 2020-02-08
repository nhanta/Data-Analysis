from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import collections


def read_data():
    #import pdb; pdb.set_trace()
    data = pd.read_csv (file_name).dropna().head(100)
    training_data = data.sample(frac = 0.8, random_state = 234)
    testing_data = data.drop(training_data.index)

    return(training_data, testing_data)

def get_variable_distribution(i):
    
    value = []
    value_dist = []
    
# Using kstest to identify random variable distribution type
    
    norm_distr = stats.kstest (read_data()[0].iloc[:, i], 
                            "norm", stats.norm.fit(read_data()[0].iloc[:, i])
                            )
    expon_distr = stats.kstest (read_data()[0].iloc[:, i],
                            "expon", stats.expon.fit(read_data()[0].iloc[:, i])
                            )
    gamma_distr = stats.kstest (read_data()[0].iloc[:, i], 
                            "gamma", stats.gamma.fit(read_data()[0].iloc[:, i])
                            )
                            
    
    min_error = min (norm_distr[0], expon_distr[0], gamma_distr[0])
    
    if norm_distr[0] == min_error and norm_distr[1] > alpha:
        value = ['norm', norm_distr[0], norm_distr[1]]
        
    elif expon_distr[0] == min_error and expon_distr[1] > alpha:
        value = ['expon', expon_distr[0], expon_distr [1]]
        
    elif gamma_distr[0] == min_error and gamma_distr[1] > alpha:
        value = ['gamma', gamma_distr [0], gamma_distr [1]]
    
# To identify poisson distribution by chisquare method

    counter_pois = collections.Counter(read_data()[0].iloc[:,i]) 

    df_pois = pd.DataFrame(list(counter_pois.items()), columns = ['arrivals', 'actual_frequency_f0'])  
    

    ld = np.mean (read_data()[0].iloc[:, i])
    df_pois['probability'] = pd.Series(stats.poisson.pmf(df_pois.arrivals, ld))
    theoretical_frequency = read_data()[0].shape[0] * df_pois.probability
    df_pois['theoretical_frequency'] = pd.Series(theoretical_frequency)

    pois_test = np.sum(
        np.square(df_pois.actual_frequency_f0 - df_pois.theoretical_frequency)/df_pois.theoretical_frequency
        )
    chisq_pois = stats.chi2.ppf(1-alpha, df_pois.shape[0] - 1 -1)
            
# To identify binomial distribution by chisquare method

    counter_binomial = collections.Counter(read_data()[0].iloc[:,i]) 

    df_binom = pd.DataFrame(list(counter_binomial.items()), columns = ['success', 'actual_frequency_f0'])   

    pi = np.mean (read_data()[0].iloc[:, i])/len(read_data()[0].iloc[:, i])
    df_binom['probability'] = pd.Series(stats.binom.pmf(df_binom.success, len(df_binom.success), pi
                                                   )
                                    )
    
    theoretical_frequency = read_data()[0].shape[0] * df_binom.probability
    df_binom['theoretical_frequency'] = pd.Series(theoretical_frequency)

    binom_test = np.sum(
        np.square(df_binom.actual_frequency_f0 - df_binom.theoretical_frequency)/df_binom.theoretical_frequency
        )
    chisq_binom = stats.chi2.ppf(1-alpha, df_binom.shape[0] - 1 -1)

 # Choise poisson distribution and binomial distribution
    
    min_dist = min(pois_test, binom_test)

    if pois_test == min_dist and pois_test < chisq_pois:
        value_dist = ['poisson']
    elif binom_test == min_dist and binom_test < chisq_binom:
        value_dist = ['binomial']
            
# Values to choice type of distribution    
    val = [value, value_dist]        
        
    return (val)
    
class Model_dataframe (object):
    
    # Print type of distribution of variable
    
    def print_variable_dataframe (self):
        
        for i in range (splitting_point +1):
            if get_variable_distribution(i)[0] != [] and get_variable_distribution(i)[0][0] == 'norm':
                print ( 
                    read_data()[0].columns.values[i], "Variable is normal distribution with D = %s, p_value = %s" 
                    % (get_variable_distribution(i)[0][1], get_variable_distribution(i)[0][2])
                    )
            elif get_variable_distribution(i)[0] != [] and get_variable_distribution(i)[0][0] == 'expon':
                print ( 
                    read_data()[0].columns.values[i], "Variable is exponential distribution with D = %s, p_value = %s" 
                    % (get_variable_distribution(i)[0][1], get_variable_distribution(i)[0][2])
                    )
            elif get_variable_distribution(i)[0] != [] and get_variable_distribution(i)[0][0] == 'gamma':
                print ( 
                    read_data()[0].columns.values[i], "Variable is gamma distribution with D = %s, p_value = %s" 
                    % (get_variable_distribution(i)[0][1], get_variable_distribution(i)[0][2])
                    )
            elif get_variable_distribution(i)[1] == ['poisson']:
                print ( 
                    read_data()[0].columns.values[i], "Variable is poisson distribution" 
                    )
            elif get_variable_distribution(i)[1] == ['binomial']:
                print ( 
                    read_data()[0].columns.values[i], "Variable is binom distribution" 
                    )
                          
    # To choose explanatory variable to add to model 
    def get_dataframe (self):
        
        for i in range(splitting_point + 1, read_data()[0].shape[1]): 
        
                df[read_data()[0].columns.values[i]] = pd.Categorical(read_data()[0].iloc[:, i])
                
        # Using cortest to identify affection among numeric variables  
        for i in range (splitting_point):
            for j in range (splitting_point):

                cor = stats.pearsonr (read_data()[0].iloc[:, i], read_data()[0].iloc[:, j])
                
                if cor[0] >= 0.8 and i<j:   
                    df[read_data()[0].columns.values[i] + "*" + read_data()[0].columns.values[j]] = read_data()[0].iloc[:, i]* read_data()[0].iloc[:, j]    

# Using cortest to identify correlation between numeric variable and response variable
    
        for i in range (splitting_point):
        
            cor = stats.pearsonr (read_data()[0].iloc[:, i], read_data()[0].iloc[:, splitting_point])
           
            if cor[1] < 0.05:
                df[read_data()[0].columns.values[i]] = pd.Series(
                        read_data()[0].iloc[:, i])
       
        df[read_data()[0].columns.values[splitting_point]] = pd.Series(read_data()[0].iloc[:, splitting_point])                         
       
        return (df) 
            
class Selecting_model (object):
    
    def __init__(self, model):
        self.model = model
    
# Getting model if response variable is normal distribution
	
    def get_normal_distribution_model (self):
        
        l = self.model.shape[1]  
        x = np.array([self.model.iloc[: , i] for i in range(l-1)]).T
        y = np.array(self.model.iloc[:, l-1]).T
        model_1 = smf.glm ('y~x', data = self.model, family = sm.families.Gaussian()).fit()  
              
        return(model_1)      
    
    def get_log_linear_poisson_model (self):
        
        l = self.model.shape[1]  
        x = np.array([self.model.iloc[: , i] for i in range(l-1)]).T
        y = np.array(self.model.iloc[:, l-1]).T
        model_3 = smf.glm("y ~ x", data = self.model, 
                          family = sm.families.Poisson(link = sm.genmod.families.links.log)).fit()
        
        return (model_3)
    
    def get_gamma_distribution_model (self):
        
        l = self.model.shape[1]  
        x = np.array([self.model.iloc[: , i] for i in range(l-1)]).T
        y = np.array(self.model.iloc[:, l-1]).T
        model_4 = smf.glm(formula = 'y ~ x', data = self.model, family = sm.families.Gamma()).fit()
        
        return (model_4)
        
    def get_binomial_distribution_model (self):
        
        l = self.model.shape[1]  
        x = np.array([self.model.iloc[: , i] for i in range(l-1)]).T
        y = np.array(self.model.iloc[:, l-1]).T
        model_5 = smf.glm(formula = 'y ~ x', data = self.model, family = sm.families.Binomial()).fit()
        
        return (model_5)
    
    def get_expon_distribution_model (self):
        
        l = self.model.shape[1]  
        x = np.array([self.model.iloc[: , i] for i in range(l-1)]).T
        y = np.array(self.model.iloc[:, l-1]).T
        model_6 = smf.glm(formula = 'y ~ x', data = self.model, 
                          family = sm.families.Gamma(link = sm.genmod.families.links.log)).fit()
        
        return (model_6)
        
    # Getting model after indentifying the distribution of response variable
        
    def select_model (self):
        
        if get_variable_distribution(splitting_point)[0] != [] and get_variable_distribution(splitting_point)[0][0] == 'norm':
            selected_model = self.get_normal_distribution_model()
        
        elif get_variable_distribution (splitting_point) [0] != [] and get_variable_distribution(splitting_point)[0][0] == 'gamma':  
            selected_model = self.get_gamma_distribution_model ()
            
        elif get_variable_distribution (splitting_point)[0] != [] and get_variable_distribution (splitting_point) [0][0] == "expon":
            selected_model = self.get_expon_distribution_model ()
            
        elif get_variable_distribution (splitting_point) [1] == ['poisson']:
            selected_model = self.get_log_linear_poisson_model ()
            
        elif get_variable_distribution (splitting_point) [1] == ['binomial']:
            selected_model = self.get_binomial_distribution_model ()
            
        return (selected_model)
    
class Filter_model (Model_dataframe):
    
    # Comparing aic among models to choice model
    
    def choice_model (self):
        
        max_pvalues_index = np.argmax(Selecting_model (super(Filter_model, self).get_dataframe()).select_model()._results.pvalues)
        aic = Selecting_model (super(Filter_model, self).get_dataframe()).select_model()._results.aic
        
        if super(Filter_model, self).get_dataframe().shape[1] > 2:
            
            step_model = super(Filter_model, self).get_dataframe().drop(labels = super(Filter_model, self).get_dataframe().columns.values[max_pvalues_index-1], axis = 1)
            if step_model.shape[1] > 2:    
                while aic > Selecting_model (step_model).select_model()._results.aic:
                    
                    max_pvalues_index = np.argmax(Selecting_model (step_model).select_model()._results.pvalues)
                    aic = Selecting_model (step_model).select_model()._results.aic
                    choosing_model_result = Selecting_model (step_model).select_model()
                    choosing_model = step_model
                    step_model = step_model.drop(labels = step_model.columns.values[max_pvalues_index-1], axis = 1)                                   
                    
                return ([choosing_model_result, choosing_model]) 
            else:
                return ([Selecting_model(step_model).select_model(), step_model])
        else:
            return ([Selecting_model (super(Filter_model, self).get_dataframe()).select_model(), super(Filter_model, self).get_dataframe()])
            
    
    def eval_model(self):  
        # get model
        get_model = Filter_model().choice_model()[1].columns.values
        test_data = read_data()[1] 
        x_predict = test_data[get_model]
        l = x_predict.shape[0] - 1
        x_predict.drop([l], inplace = True)  
        
        # get test data
        
        y_obs = test_data[splitting_point]
        y_predict = self.choice_model()[0]._results.predict ({'x' : x_predict}).values[0]
        
        # import pdb; pdb.set_trace()
        #print(predict_list)
        loss = ((y_obs - y_predict) * 2 )/len(y_predict)
        diff = y_obs - y_predict 
        print("+) LOSS: " + str(loss))
        print("")
        
        print("+) DETAIL TABLE: ")
        print("")
        eval_detail = pd.DataFrame({'Y': y_obs, 'PREIDICT': y_predict, 'DIFF': diff})
        print(eval_detail)
    

    def result_predict(self):
        # get model
        get_model = Filter_model().choice_model()[1].columns.values
        
        model = []
        for i in range(len(get_model)):
            if len(get_model[i].split("*")) >= 2:
                model.append(get_model[i].split("*"))
            else:
                model.append(get_model[i])
        
        print("")
        print("+) MODEL: " + str(get_model) )
        print("")
        
        #inputting_x = np.fromstring(input("Inputting variable (separate each element by space): "), dtype=np.float, sep=' ')
        inputting_x = []
        print("+) INPUT FOR PREDICTION: ")
        for i in range(len(model[:-1])):
            if type(model[i]) == list:
                temp = 1
                for j in range(len(model[i])):
                    temp *= float(input(model[i][j] + ": "))
                inputting_x.append(temp)
            else:
                inputting_x.append(float(input(model[i] + ": ")))
        
        #import pdb; pdb.set_trace()
        x = np.array([inputting_x])
        predict = self.choice_model()[0]._results.predict ({'x' : x}).values[0]
        print("")
        print("=> PREDICTION RESULT: " + str(predict))
    
        # Predict next time
        print("")
        next_predict = str(input("DO YOU WANT TO PREDICT ON MORE TIME (y/n) : "))
  

        while next_predict != 'y' and next_predict != 'n':
            print("")
            print("WARNING: PLEASE INPUT: 'y' OR 'n'")
            print("")
            next_predict = str(input("DO YOU WANT TO PREDICT ON MORE TIME (y/n) : "))
            
        
        while next_predict == 'y':
            print("")
            inputting_x = []
            print("+) INPUT FOR PREDICTION: ")
            for i in range(len(model[:-1])):
                if type(model[i]) == list:
                    temp = 1
                    for j in range(len(model[i])):
                        temp *= float(input(model[i][j] + ": "))
                    inputting_x.append(temp)
                else:
                    inputting_x.append(float(input(model[i] + ": ")))
                    
            x = np.array([inputting_x])
            predict = self.choice_model()[0]._results.predict ({'x' : x}).values[0]
            print("")
            print("=> PREDICTION RESULT: " + str(predict))
        
            # Predict next time
            print("")
            next_predict = str(input("DO YOU WANT TO PREDICT ONE MORE TIME (y/n) : "))    

            while next_predict != 'y' and next_predict != 'n':
                print("")
                print("WARNING: PLEASE INPUT: 'y' OR 'n'")
                print("")
                next_predict = str(input("DO YOU WANT TO PREDICT ONE MORE TIME? (y/n) : "))
    
    
        if next_predict == 'n':
            print("")
            print("")
            print("********************************** FINISHED **********************************")
            print("")

        
if __name__ == '__main__':
    # Before splitting point is numeric variable, after that one is factor variable
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|              -----     AUTO MODELLING FOR STATISTIC    -----               |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    
    print("********************************* INPUT INFO *********************************")
    print("")
    file_name = input("Input file name: ")
    alpha = float(input ("Input alpha: "))
    splitting_point = int(input ("Input splitting point : "))
    print("(Before splitting point is numeric variable, after that one is factor variable)")
    print("")
    df = pd.DataFrame () 
    
    print("")
    print("*********************************** RESULT ***********************************")
    print("")
    
    # Print name of variables in filter model
    print ("MODEL: " + str(Filter_model().choice_model()[1].columns.values))
    print("")

    # Print parameters of filter model      
    #print(Filter_model().choice_model()[0]._results.params)

    # Print the summary of filter model
    print("SUMMARY MODEL: ")
    print("")
    print(Filter_model().choice_model()[0].summary())

    # Print AIC of the filter model
    print("")
    print("AIC: " + str(Filter_model().choice_model()[0]._results.aic))
    # Print evaluation model

    print("")
    print("EVALUATION MODEL: ")
    print("")
    Filter_model().eval_model()

    # Pridicting results of response variable
    print("")
    print("")
    print("********************************** PREDICT **********************************")
    print("")
    Filter_model().result_predict()


    




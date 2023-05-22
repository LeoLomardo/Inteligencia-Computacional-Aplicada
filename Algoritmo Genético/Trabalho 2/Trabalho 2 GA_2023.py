# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:38:06 2021

@author: Karla
"""

import numpy as np
import math
import locale
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Callbacks
from geneticalgorithm2 import Population_initializer
from geneticalgorithm2 import Generation, AlgorithmParams 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from OptimizationTestFunctions import Rastrigin, plot_3d

#############################################################################################
# Função do problema

def f(X):

    dim=len(X)         

    OF=0
    for i in range (0,dim):
        OF+=(X[i]**2)-10*math.cos(2*math.pi*X[i])+10

    return OF
#############################################################################################


#limites do espaço de busca para as variáveis
varbound=np.array([[-5.12,5.12]]*2)

# parâmetros do GA
max_num_iteration = 25  # número de gerações
population_size = 250
mutation_probability = 1.0
elit_ratio=0.005 # percentual de individuos preservados na próxima geracao
crossover_probability = 1.0
parents_portion = 0.3# zero, significa que toda a população é 
                    # preenchida com as soluções recém-geradas
crossover_type= 'one_point'

selection_type='roulette'

num_experimentos = 50 #número de rodadas sucessivas


                   
algorithm_param = {'max_num_iteration': max_num_iteration,\
                   'population_size':population_size,\
                   'mutation_probability':mutation_probability,\
                   'elit_ratio': elit_ratio,\
                   'crossover_probability': crossover_probability,\
                   'parents_portion': parents_portion,\
                   'crossover_type':crossover_type,\
                   'max_iteration_without_improv':None,
                   'mutation_type': 'uniform_by_center',
                   'selection_type': selection_type                     
                  }

model=ga(function=f,\
            dimension=2,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)
    


simulacoes=[]
for simu in range(0, num_experimentos+1):
    
    print()
    print('-------------------------------------------------------------------')
    print('Experimentos número = ', simu)
    model.run(no_plot = False,
                  disable_progress_bar = False,
                  set_function = None, 
                  apply_function_to_parents = False, 
                  start_generation = {'variables':None, 'scores': None},
                  studEA = True,
                  mutation_indexes = None,
                  init_creator = None,
                  init_oppositors = None,
                  duplicates_oppositor = None,
                  remove_duplicates_generation_step = 1,
                  revolution_oppositor = None,
                  revolution_after_stagnation_step = None,
                  revolution_part =0,
                  population_initializer = Population_initializer(select_best_of = 1, local_optimization_step = 'never', local_optimizer = None),
                  stop_when_reached = None,
                  callbacks=[
                  Callbacks.SavePopulation('callback_pop_example', save_gen_step=1, file_prefix='constraints'),
                  Callbacks.PlotOptimizationProcess('callback_plot_example', save_gen_step=300, show = False, main_color='red', file_prefix='plot')
                  ],                            
                  middle_callbacks = [],
                  time_limit_secs = None, 
                  save_last_generation_as = None,
                  seed = None
                  )
    
        
    # title = f"Busca do ótimo para {type(f).__name__}"
    # model.plot_results(title = title, save_as = f"{title}.png", main_color = 'green')
    model.plot_generation_scores()
	
    convergence=model.report
    print("melhores individuos por geração",convergence)
    
    simulacoes.append(convergence)

mean_simulation=[]    
    
for i in range(0, max_num_iteration):
    soma_sim_geracao=0
    for j in range(0, num_experimentos):
        soma_sim_geracao=soma_sim_geracao+simulacoes[j][i]
    mean_simulation.append(soma_sim_geracao/num_experimentos)


print('------------------------------------------------------------------------')    
print('Valores médios dos melhores por Geração:')    
print(str(mean_simulation).replace('.',','))

print('------------------------------------------------------------------------')    
print('Média valores médios dos melhores por Geração:')    
media_medios = np.mean(mean_simulation)
print(str(media_medios).replace('.',','))

print('------------------------------------------------------------------------')    
print('Menor valor médio:')    
print(str(mean_simulation[max_num_iteration - 1]).replace('.',','))

contador = 3

fig1, ax1 = plt.subplots()
ax1.set_title('Media dos Melhores por Geração')
ax1.boxplot(mean_simulation)
plt.savefig(f"/path/to/your/drive/folder/Imagens/padrao{0}_1.png")
plt.show()

plt.plot(mean_simulation, label='Média dos Melhores por Geração')
plt.legend(loc='upper right')
plt.savefig(f"/path/to/your/drive/folder/Imagens/padrao{0}_2.png")
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

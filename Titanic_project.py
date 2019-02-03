
# coding: utf-8

# #### Fundamentos Data Science I 
# _Udacity Jul/2018 - Juliano M Mendes_
# 
# # Projeto Titanic: Investigando uma base de dados kaggle
# 
# ### Visão Geral do Projeto
# 
# Neste projeto, será analisado um conjunto de dados de passageiros do naufrágio do RMS Titanic e, em seguida, serão comunicadas conclusões sobre as análises realizadas. As bibliotecas utilizadas serão: NumPy, Pandas e Matplotlib.
# 
# ### Resumo Histórico
# O naufrágio do RMS Titanic é um dos mais notórios naufrágios da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou depois de colidir com um iceberg, matando 1502 de 2224 passageiros e tripulantes. Esta tragédia sensacional chocou a comunidade internacional e levou a melhores normas de segurança para os navios.
# 
# ### Objetivo
# 
# **O objetivo deste projeto será responder aos seguintes questionamentos sobre os dados:**
# 
# 1 - Quais foram os fatores que fizeram com que algumas pessoas mais propensas a sobreviver?  
# 2 - Será que a classe mais alta teve mais chances de sobrevivência que a mais baixa?  
# 3 - A proporção de sobrevivência da primeira classe foi superior em relação à segunda e a terceira classe?  
# 4 - Há correlação direta entre os que sobreviveram e seus gêneros?  
# 5 - Qual a proporção de sobreviventes e mortos por gênero?  
# 6 - Mulheres e crianças salvaram-se primeiro?  
# 7 - Há correlação direta entre o valor pago na passagem e idade?  
# 8 - Quem são os passageiros que não tiveram a tarifa cobrada? Eles sobreviveram?  
# 
# **Tendo como resultado de aprendizado:**
# 
# - Visão ampla dos passos envolvidos em um processo de análise de dados típico;
# - Realização de questionamentos que podem ser respondidos por um conjunto de dados;
# - Capacidade de investigar problemas em um conjunto de dados e confrontrar os dados em um formato que se possa usar;
# - Adquirir prática em comunicar os resultados de sua análise;
# - Ser capaz de utilizar as operações vetorizadas no NumPy e Pandas para acelerar código de análise de dados;
# - Possuir familiaridade com objetos, séries e o banco de dados Pandas, para acesso aos dados de forma mais conveniente;
# - Saber utilizar a biblioteca Matplotlib para produzir gráficos simples e objetivos.
# 
# ***
# 
# ### Obtendo o conjunto de dados
# 
# > **[Dados do Titanic](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59e4fe3d_titanic-data-6/titanic-data-6.csv)** - contém dados demográficos e informações de 891 dos 2.224 passageiros e tripulantes a bordo do Titanic. Você pode ver uma descrição deste conjunto de dados no site do [Kaggle](https://www.kaggle.com/c/titanic/data#), de onde os dados foram tirados.

# ## Carregando os dados com Pandas (load data)

# In[1]:


# Importando os pacotes e carregando os arquivos de titanic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df_titanic = pd.read_csv('titanic-data-6.csv')


# ## Exploring Titanic Dataset (EDA with Visualizations)
# 
# ### Análise Exploratória de dados do Titanic###
# 
# Nesta seção será analisado o conjunto de dados Titanic

# ### Data Dictionary
# 
# <table align="left">
#   <tr>
#     <th>Column</th>
#     <th>Definition</th>
#     <th>Key</th>
#   </tr>
#   <tr>
#     <td>Survival</td>
#     <td>Survival</td>
#     <td>0 = No, 1 = Yes</td>
#   </tr>
#   <tr>
#     <td>Pclass</td>
#     <td>Ticket class</td>
#     <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
#   </tr>
#   <tr>
#     <td>Sex</td>
#     <td>Sex</td>
#     <td>male, famale</td>
#   </tr>
#   <tr>
#     <td>Age</td>
#     <td>Age in years</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Sibsp</td>
#     <td># of siblings / spouses aboard the Titanic</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Parch</td>
#     <td># of parents / children aboard the Titanic </td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Ticket</td>
#     <td>icket number</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Fare</td>
#     <td>Passenger fare</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Cabin</td>
#     <td>Cabin number </td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Embarked</td>
#     <td>Port of Embarkation</td>
#         <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
#   </tr>
# </table>

# ***
# ### Variable Notes
# 
# A. **Pclass**: A proxy for socio-economic status (SES)  
#   - 1st = Upper  
#   - 2nd = Middle    
#   - 3rd = Lower  
# 
# B. **Age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5  
# 
# C. **Sibsp**: The dataset defines family relations in this way:  
#   - Sibling = brother, sister, stepbrother, stepsister  
#   - Spouse = husband, wife (mistresses and fiancés were ignored)
#   
# D. **Parch**: The dataset defines family relations in this way:  
#   - Parent = mother, father  
#   - Child = daughter, son, stepdaughter, stepson  
#   - Some children travelled only with a nanny, therefore parch=0 for them.  
#   
# #### Esta amostra possui `891` registros obtido do arquivo `titanic-data-6.csv`

# ### Organização do conjunto de dados
# Informações gerais do conjunto de dados e estatísticas preliminares

# In[2]:


# Verificando as primeiras informações do conjunto de dados
df_titanic.head()


# In[3]:


# Checando a quantidade de linhas e colunas
df_titanic.shape


# In[4]:


# Removendo a coluna PassengerID que não será utilizada
df_titanic.drop(['PassengerId'], axis=1, inplace=True)


# In[5]:


# Calculando quais colunas possuem nulos e suas quantidades.
df_titanic.isnull().sum()


# In[6]:


# Resumo de informações do Dataframe
df_titanic.info()


# In[7]:


# Estatísticas gerais do Dataframe
df_titanic.describe()


# In[8]:


df_titanic.duplicated().sum()


# In[9]:


df_titanic['Age'].isnull().sum()


# In[246]:


# Grafico de distribuição de idades dos passageiros
df_titanic['Age'].plot(kind='hist', title='Faixa etária dos passageiros considerando \n valores nulos');


# In[248]:


# Usando mean para preencher as idades faltantes
mean = df_titanic['Age'].mean()
df_titanic['Age'].fillna(mean, inplace=True)

# Grafico de distribuição de idades dos passageiros com o preenchimento dos campos nulos com a média de idades
df_titanic['Age'].plot(kind='hist', title='Faixa etária dos passageiros considerando \n a média de idades');


# In[12]:


# Verificando se os nulos foram preenchidos
df_titanic['Age'].isnull().sum()


# In[13]:


# Estatísticas do Dataframe: Média de idade
df_titanic['Age'].mean()


# In[15]:


# Estatísticas do Dataframe: Média de idade dos sobreviventes
df_titanic[df_titanic['Survived']== 1].mean()['Age']


# In[16]:


# Estatísticas do Dataframe: Média de idade dos mortos
df_titanic[df_titanic['Survived']== 0].mean()['Age']


# In[17]:


# Estatísticas do Dataframe: Média de idade dos homens
df_titanic[df_titanic['Sex']== 'male'].mean()['Age']


# In[500]:


# Estatísticas do Dataframe: Média de idade das mulheres
df_titanic[df_titanic_cleaned['Sex']== 'female'].mean()['Age']


# In[249]:


# Deixando o dataframe apenas com as colunas que interessam para as próximas análises
df_titanic_subset = df_titanic.iloc[:, np.r_[0:5]]
df_titanic_cleaned = df_titanic_subset.drop(columns=['Name'], axis=1)
df_titanic_cleaned.head()


# In[65]:


# Totais para Homens
df_titanic_cleaned[df_titanic_cleaned['Sex']=='male'].count()


# In[66]:


# Totais para Mulheres
df_titanic_cleaned[df_titanic_cleaned['Sex']=='female'].count()


# In[250]:


# Organizando os Dataframes por gênero
df_titanic_m = df_titanic_cleaned[df_titanic_cleaned['Sex']=='male']
df_titanic_f = df_titanic_cleaned[df_titanic_cleaned['Sex']=='female']


# ## Data Analysis e conclusions
# 
# ### Analisando os dados e tirando conclusões
# 
# A partir desta seção serão realizadas as análises mais aprofundadas no dataset, e serão tiradas algumas conclusões sobre os resultados obtidos.

# ### Sobreviventes e Mortos por gênero (Homens e Mulheres)

# In[24]:


# Numeros de Sobrevivência de Homens
df_titanic_m.groupby('Survived').count()['Sex']


# In[25]:


# Proporção de Sobrevivência de Homens
df_titanic_m.groupby('Survived').count()['Sex'] / df_titanic_m.count()['Sex'] * 100


# In[251]:


# Grafico de barras
df = (df_titanic_m.groupby('Survived').count()['Sex'] / df_titanic_m.count()['Sex'] * 100).tolist()
plt.bar([1,2], df, color=['darkred','steelblue'])
plt.xticks([1,2], ['Não','Sim'])
plt.title('Percentual de Sobreviventes Homens')
plt.xlabel('Sobreviveu?')
plt.ylabel('Percentual');


# De acordo com o gráfico acima, observamos que 81% dos *homens* (passageiros do gênero masculino) não conseguiram sobreviver, enquanto apenas 18.8% sobreviveram, considerando adultos e crianças.

# In[27]:


# Numeros de Sobrevivência de Mulheres
df_titanic_f.groupby('Survived').count()['Sex']


# In[28]:


# Proporção de Sobrevivência de Mulheres
df_titanic_f.groupby('Survived').count()['Sex'] / df_titanic_f.count()['Sex'] * 100


# In[252]:


# Grafico de barras
df = (df_titanic_f.groupby('Survived').count()['Sex'] / df_titanic_f.count()['Sex'] * 100).tolist()
plt.bar([1,2], df, color=['darkred','steelblue'])
plt.xticks([1,2],['Não','Sim'])
plt.title('Percentual de Sobreviventes Mulheres')
plt.xlabel('Sobreviveu?')
plt.ylabel('Percentual');


# De acordo com o gráfico acima, observamos que 25.7% das *mulheres* (passageiras do gênero feminino) não conseguiram sobreviver, enquanto a maioria 74.2% sobreviveu, considerando entre adultos e crianças. 

# In[68]:


# Proporção de sobreviventes homens do total de passageiros (homens e mulheres)
df_titanic_m.groupby('Survived').count()['Sex'] / df_titanic.count()['Sex'] * 100


# In[69]:


# Proporção de sobreviventes mulheres do total de passageiros (homens e mulheres)
df_titanic_f.groupby('Survived').count()['Sex'] / df_titanic.count()['Sex'] * 100


# In[253]:


# Organizando o Dataframe de Sobreviventes
df_survived = df_titanic.query('Survived == 1')
df_survived['Sex'].count()


# In[254]:


# Proporção de Sobreviventes por Gênero
df_survived.groupby('Sex').count()['Survived'] / df_titanic.count()['Sex'] * 100


# In[77]:


# Proporção de sobreviventes sobre todos os passageiros do navio
df_survived.groupby('Survived').count()['Sex'] / df_titanic.count()['Sex'] * 100


# In[98]:


lst_survived = [df_titanic_m.groupby('Survived').count()['Sex'][1], df_titanic_f.groupby('Survived').count()['Sex'][1]]


# In[256]:


# Comparativo de sobreviventes por Gênero (Homens x Mulheres)
plt.bar([1,2], lst_survived, color=['steelblue','slategrey'])
plt.xticks([1,2],['Homens','Mulheres'])
plt.title('Comparativo de sobreviventes por Gênero \n (Homens e Mulheres)')
plt.xlabel('Sexo')
plt.ylabel('Total');

# Sobreviventes por Gênero
#df_titanic_s['Sex'].value_counts().plot(kind='bar');


# ### Conclusões sobre a relação entre sobrevivência e o gênero dos passageiros
# Concluímos que a maior proporção de sobreviventes foram as mulheres. A taxa de sobrevivência das mulheres ficou um pouco acima do dobro da taxa de sobreviventes homens:
# 
# #### Estatísticas:
# - Total de passageiros = 891
# - Total de sobreviventes = 342 (38%)
# - Sobreviventes Homens = 109 (12%)
# - Sobreviventes Mulheres = 233 (26%)
# 
# Podemos afirmar até aqui, de acordo com os dados disponíveis analisados, que o gênero foi um dos fatores decisivos na sobrevivência ao naufrágio.

# ### Sobreviventes e Mortos por Classe

# In[263]:


# Preparação dos Dataframes de sobreviventes e mortos
df_titanic_s = df_titanic_cleaned[df_titanic_cleaned['Survived']== 1]
df_titanic_m = df_titanic_cleaned[df_titanic_cleaned['Survived']== 0]


# In[264]:


# Verificando o total de passageiros por Classe
df_titanic.groupby('Pclass').count()['Survived']


# In[265]:


# Sobreviventes por Classe
df_titanic_s.groupby('Pclass').count()['Survived']


# In[266]:


# Mortos por Classe
df_titanic_m.groupby('Pclass').count()['Survived']


# In[267]:


# Proporção de Sobreviventes por Classe
df_titanic_s.groupby('Pclass').count()['Survived'] / df_titanic_s.count()['Survived'] * 100


# Nota: Os que mais sobreviveram estavam na 1.ª classe (39.8%)

# In[268]:


# Proporção de Mortos por Classe
df_titanic_m.groupby('Pclass').count()['Survived'] / df_titanic_m.count()['Survived'] * 100


# Nota: Os que mais morreram estavam na 3.ª classe (67.8%)

# In[269]:


# Grafico de barras
df = (df_titanic_s.groupby('Pclass').count()['Survived'] / df_titanic_s.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1a','2a', '3a'])
plt.title('Proporção de sobreviventes por Classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# De acordo com o gráfico acima, dentre os passageiros que sobreviveram, os da 1.ª classe (sem distinção por gênero) foram os que mais sobreviveram. Observa-se que ***39.8%*** de 1.ª classe conseguiram sobreviver, enquanto ***25.4%*** da 2.ª classe e ***34.8%*** de sobreviventes da 3.ª classe, proporcionalmente em suas classes. De acordo com esta primeira análise, observamos que não houve essencialmente como fator decisivo de sobrevivência, a relação de classe do passageiro. Porém, em termos proporcionais, os passageiros da 1.ª classe tiveram uma melhor taxa de sobrevivência.

# In[270]:


# Proporção de Sobreviventes por Classe do total de passageiros (homens e mulheres)
df_titanic_s.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100


# Nota: A 1.ª classe se mantém com o melhor índice de sobrevivência (15.3%)

# In[271]:


# Grafico de barras
df = (df_titanic_s.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª', '3.ª'])
plt.title('Proporção de sobreviventes em cada Classe \n pelo Total de passageiros')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# Observa-se que mesmo em relação ao total de passageiros, as proporções são basicamente mantidas, confirmando a análise acima de que o fator classe não foi tão determinante para a sobrevivência de passageiros. Tendo é claro, percentuais inferiores de sobreviventes em relação ao total, do que proporcionalmente dentro de cada classe.

# In[273]:


# Grafico de barras
df = (df_titanic_m.groupby('Pclass').count()['Survived'] / df_titanic_m.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª','3.ª'])
plt.title('Proporção de mortes por Classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# Em relação às mortes, cerca de **68%** dos passageiros da 3.ª classe não conseguiram sobreviver, enquanto que as taxas das classes superiores, observa-se significativa redução de mortes (**17.6%** na 2.ª classe e **14.5%** na 1.ª classe).
# 
# ** Podemos concluir que termos proporcionais, a 1.ª e a 3.ª classes mantiveram seus % aproximados, porém quando observamos os números totais, a quantidade de mortos foi bem superior na 3.ª classe. **

# In[274]:


# Grafico de barras
df = (df_titanic_m.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª','3.ª'])
plt.title('Proporção de mortes por Classe \n pelo Total de passageiros')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# No gráfico acima observa-se disposição proporcional semelhante em relação às mortes por classe (gráfico anterior).
# 
# ** Conclui-se que a classe foi um fator determinante nas mortes dos passageiros. As classes mais altas tiveram as menores taxas entre os mortos em relação a classe mais baixa. **

# In[275]:


# Grafico comparativo de sobreviventes x mortos por classe
df_s = (df_titanic_s.groupby('Pclass').count()['Survived'] / df_titanic_s.count()['Survived'] * 100).tolist()
df_m = (df_titanic_m.groupby('Pclass').count()['Survived'] / df_titanic_m.count()['Survived'] * 100).tolist()
plt.bar([1,4,7], df_s, color=['steelblue'], label='Sobreviventes')
plt.bar([1.8,4.8,7.8], df_m, color=['darkred'], label='Mortos')
plt.xticks([1.4,4.4,7.4],['1a','2a','3a'])
plt.legend()
plt.title('Comparativo de sobreviventes x mortos por classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# In[276]:


# Grafico comparativo de sobreviventes x mortos por classe do total de passageiros (homens e mulheres)
df_s = (df_titanic_s.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100).tolist()
df_m = (df_titanic_m.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100).tolist()
plt.bar([1,4,7], df_s, color=['steelblue'], label='Sobreviventes')
plt.bar([1.8,4.8,7.8], df_m, color=['darkred'], label='Mortos')
plt.xticks([1.4,4.4,7.4],['1a','2a','3a'])
plt.legend()
plt.title('Comparativo de sobreviventes x mortos por classe \n em relação ao total de passageiros')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# ### Conclusões sobre a relação entre sobrevivência e morte dos passageiros por Classe
# Observa-se que a maior proporção de sobreviventes estavam presente na 1.ª classe, e em contrapartida, a maioria das mortes ocorreu com passageiros da 3.ª classe. A relação entre as taxas de sobrevivência e morte da 2.ª classe ficaram relativamente próximas.
# 
# #### Estatísticas:
# - Total de passageiros = 891
# - Total de sobreviventes = 342 (38%)
# ***
# - Total de passageiros na 1.ª Classe = 216
# - Total de passageiros na 2.ª Classe = 184
# - Total de passageiros na 3.ª Classe = 491
# ***
# - Sobreviventes na 1.ª Classe = 136 (39.7%)
# - Sobreviventes na 2.ª Classe =  87 (25.4%)
# - Sobreviventes na 3.ª Classe = 119 (34.8%)
# ***
# - Mortos na 1.ª Classe =  80 (14.5%)
# - Mortos na 2.ª Classe =  97 (17.6%)
# - Mortos na 3.ª Classe = 372 (67.8%)
# 
# ** De acordo com os dados analisados, podemos afirmar que dentro destes parâmetros, a classe foi um dos fatores determinantes para a sobrevivência dos passageiros. **

# ### Sobreviventes e Mortos por Gênero (Homens e Mulheres) e Classe

# In[277]:


# Preparação dos dados de sobreviventes e mortos Homens
df_titanic_s_h = df_titanic_s[df_titanic_s['Sex'] == 'male']
df_titanic_m_h = df_titanic_m[df_titanic_m['Sex'] == 'male']

# Preparação dos dados de sobreviventes e mortos Mulheres
df_titanic_s_m = df_titanic_s[df_titanic_s['Sex'] == 'female']
df_titanic_m_m = df_titanic_m[df_titanic_m['Sex'] == 'female']


# In[278]:


# Sobreviventes Homens por Classe
df_titanic_s_h.groupby('Pclass').count()['Survived']


# In[279]:


# Mortos Homens por Classe
df_titanic_m_h.groupby('Pclass').count()['Survived']


# In[280]:


# Proporção de Sobreviventes Homens por Classe
df_titanic_s_h.groupby('Pclass').count()['Survived'] / df_titanic_s_h.count()['Survived'] * 100


# In[281]:


# Proporção de mortos Homens por Classe
df_titanic_m_h.groupby('Pclass').count()['Survived'] / df_titanic_m_h.count()['Survived'] * 100


# In[285]:


# Grafico de barras
df = (df_titanic_s_h.groupby('Pclass').count()['Survived'] / df_titanic_s_h.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª', '3.ª'])
plt.title('Proporção de Homens sobreviventes por Classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# Observamos que dentre os passageiros **homens** e apenas sobre os números daqueles que sobreviveram, que os da 3.ª classe tiveram um valor proporcional mais alto de sobrevivência (**43.1%**), enquanto que ***15.6%*** na 2.ª classe e ***41.3%*** na 1.ª classe. 
# 
# Dentro deste cenário, observamos que não podemos afirmar como fator decisivo de sobrevivência, a relação de classe e passageiros do gênero masculino. Verificamos que para os homens, olhando apenas sobre este prisma, a classe não foi um fator determinante para a sobrevivência.

# In[118]:


df_titanic_s_h.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100


# In[287]:


# Grafico de barras
df = (df_titanic_s_h.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª', '3.ª'])
plt.title('Proporção de Homens sobreviventes por Classe \n pelo Total de passageiros')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# No gráfico acima observa-se uma disposição proporcional semelhante na relação de sobreviventes **homens** por classe (gráfico anterior). Mantém-se nesta segunda visão, que os sobreviventes da 3.ª classe são em proporção, superiores aos sobreviventes das demais classes.

# In[288]:


# Grafico comparativo de homens sobreviventes x mortos por classe
df_s = (df_titanic_s_h.groupby('Pclass').count()['Survived'] / df_titanic_s_h.count()['Survived'] * 100).tolist()
df_m = (df_titanic_m_h.groupby('Pclass').count()['Survived'] / df_titanic_m_h.count()['Survived'] * 100).tolist()
plt.bar([1,4,7], df_s, color=['steelblue'], label='Sobreviventes')
plt.bar([1.8,4.8,7.8], df_m, color=['darkred'], label='Mortos')
plt.xticks([1.4,4.4,7.4],['1.ª','2.ª','3.ª'])
plt.legend()
plt.title('Comparativo de homens sobreviventes x mortos \n por classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# ### Conclusões sobre a relação entre sobrevivência e morte dos passageiros homens
# Nesta visao, fica fácil perceber que mesmo tendo um grande número de sobreviventes na 3.ª classe, a 1.ª classe teve melhor taxa de sobrevivencia se comparando com as mortes. A 2.ª classe, mesmo tendo o número de mortos maior que o de sobrevivenetes, ainda se mantém em posição melhor se compararmos os números da 3.ª classe, que teve um numero muito mais expressivo de mortes.
# 
# Apesar de nas visoes anteriores, proporcionalmente dentro da 3.ª classe, o número de sobreviventes em relação às demais classes foi maior, aqui fica claro que a classe entre os homens foi um fator determinante para a sobrevivencia.
# 
# Observa-se que a 2.ª classe teve o pior resultado proporcional, tendo em vista que o número de sobreviventes e mortos ficaram próximos. Já a 1.ª classe teve o melhor resultado se comparamos os números de sobreviventes x mortos.
# 
# #### Estatísticas:
# - Total de passageiros = 891
# - Total de sobreviventes = 342 (38%)
# - Sobreviventes Homens = 109 (12%)
# ***
# - Total de passageiros na 1.ª Classe = 216
# - Total de passageiros na 2.ª Classe = 184
# - Total de passageiros na 3.ª Classe = 491
# ***
# - Sobreviventes na 1.ª Classe = 136 (39.7%)
# - Sobreviventes na 2.ª Classe =  87 (25.4%)
# - Sobreviventes na 3.ª Classe = 119 (34.8%)
# ***
# - Mortos na 1.ª Classe =  80 (14.5%)
# - Mortos na 2.ª Classe =  97 (17.6%)
# - Mortos na 3.ª Classe = 372 (67.8%)
# ***
# - Sobreviventes ***homens*** na 1.ª Classe = 45 (41.3%)
# - Sobreviventes ***homens*** na 2.ª Classe = 17 (15.6%)
# - Sobreviventes ***homens*** na 3.ª Classe = 47 (43.1%)
# ***
# - Mortos ***homens*** na 1.ª Classe =  77 (16.5%)
# - Mortos ***homens*** na 2.ª Classe =  91 (19.4%)
# - Mortos ***homens*** na 3.ª Classe = 300 (64.1%)

# In[45]:


# Sobreviventes Mulheres por Classe
df_titanic_s_m.groupby('Pclass').count()['Survived']


# Nota: A maioria das mulheres sobreviventes está na 1.ª classe. Porém, observa-se que as demais classes mativeram um número muito aproximado de sobreviventes.

# In[46]:


# Mortos Mulheres por Classe
df_titanic_m_m.groupby('Pclass').count()['Survived']


# Nota: Para as mulheres, a classe também foi um fator determinantes no número de mortes.

# In[47]:


# Proporção de Mulheres Sobreviventes por Classe
df_titanic_s_m.groupby('Pclass').count()['Survived'] / df_titanic_s_m.count()['Survived'] * 100


# In[125]:


# Proporção de Mulheres Mortas por Classe
df_titanic_m_m.groupby('Pclass').count()['Survived'] / df_titanic_m_m.count()['Survived'] * 100


# In[289]:


# Grafico de barras
df = (df_titanic_s_m.groupby('Pclass').count()['Survived'] / df_titanic_s_m.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª', '3.ª'])
plt.title('Proporção Mulheres sobreviventes por Classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# Observamos que dentre as passageiras **mulheres** que sobreviveram, aquelas da 1.ª classe foram as que proporcionalmente, mais sobreviveram (**39.1%**), enquanto ***30%*** na 2.ª classe e ***30.9%*** sobreviveram na 3.ª classe. Podemos notar que os valores proporcionalmente ficaram bem próximos, e que a classe, não foi essencialmente o fator decisivo de sobrevivência, se olharmos por esta visão.

# In[120]:


df_titanic_s_m.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100


# In[290]:


# Grafico de barras
df = (df_titanic_s_m.groupby('Pclass').count()['Survived'] / df_titanic.count()['Survived'] * 100).tolist()
plt.bar([1,2,3], df, color=['darkslategrey','steelblue', 'slategrey'])
plt.xticks([1,2,3],['1.ª','2.ª', '3.ª'])
plt.title('Proporção de Mulheres sobreviventes por Classe \n pelo Total de passageiros')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# No gráfico acima observa-se uma disposição proporcional semelhante em relação de sobreviventes de **mulheres** por classe (gráfico anterior). Verificamos nesta segunda visão, que os sobreviventes da 1.ª classe são em proporção, superiores aos sobreviventes das demais classes. Tendo a 2.ª classe o pior percentual de sobreviventes.

# In[291]:


# Grafico comparativo de mulheres sobreviventes x mortos por classe
df_s = (df_titanic_s_m.groupby('Pclass').count()['Survived'] / df_titanic_s_m.count()['Survived'] * 100).tolist()
df_m = (df_titanic_m_m.groupby('Pclass').count()['Survived'] / df_titanic_m_m.count()['Survived'] * 100).tolist()
plt.bar([1,4,7], df_s, color=['steelblue'], label='Sobreviventes')
plt.bar([1.8,4.8,7.8], df_m, color=['darkred'], label='Mortos')
plt.xticks([1.4,4.4,7.4],['1.ª','2.ª','3.ª'])
plt.legend()
plt.title('Comparativo de Mulheres sobreviventes x mortas \n por classe')
plt.xlabel('Classe')
plt.ylabel('Percentual');


# ### Conclusões sobre a relação entre sobrevivência e morte dos passageiras mulheres por Classe
# A conclusão que se chega é que proporcionalmente dentro das classes mais altas, o número de sobreviventes em relação à 3.ª classe foi mais relevante, mas não determinante. Os valores ficaram próximos proporcionalmente. 
# 
# Porém, quando observamos o comparativo de mortes nota-se que para a 3.ª classe, foi um valor substancial, o pior resultado. Muitas mulheres da 3.ª classe não conseguiram sobreviver (**89%**). Portanto para as mulheres, a classe foi um fator determinante no número de mortes, pois há uma enorme diferença entre as classes altas e a 3.ª classe.
# 
# #### Estatísticas:
# - Total de passageiros = 891
# - Total de sobreviventes = 342 (38%)
# - Sobreviventes Mulheres = 233 (26%)
# ***
# - Total de passageiros na 1.ª Classe = 216
# - Total de passageiros na 2.ª Classe = 184
# - Total de passageiros na 3.ª Classe = 491
# ***
# - Sobreviventes na 1.ª Classe = 136 (39.7%)
# - Sobreviventes na 2.ª Classe =  87 (25.4%)
# - Sobreviventes na 3.ª Classe = 119 (34.8%)
# ***
# - Mortos na 1.ª Classe =  80 (14.5%)
# - Mortos na 2.ª Classe =  97 (17.6%)
# - Mortos na 3.ª Classe = 372 (67.8%)
# ***
# - Sobreviventes ***mulheres*** na 1.ª Classe = 91 (39.1%)
# - Sobreviventes ***mulheres*** na 2.ª Classe = 70 (30.0%)
# - Sobreviventes ***mulheres*** na 3.ª Classe = 72 (30.9%)
# ***
# - Mortos ***mulheres*** na 1.ª Classe =   3 ( 3.7%)
# - Mortos ***mulheres*** na 2.ª Classe =   6 ( 7.4%)
# - Mortos ***mulheres*** na 3.ª Classe =  72 (88.9%)

# ### Sobreviventes e Mortos por Faixa Etária (Idade) 
# - Sobreviventes e Mortos por faixa etária 
# - Homens e Mulheres por Idade  
# - Relação entre a Idade e o Valor Pago na Passagem 
# - Outliers

# ### Total de Sobreviventes e Mortos
# - ```df_titanic_s['Age'].count()``` #342
# - ```df_titanic_m['Age'].count()``` #577
# 
# ### % de Sobreviventes e Mortos
# - ```df_titanic_s.groupby('Survived').count()['Age'] / df_titanic.count()['Survived'] * 100``` #38.383838
# - ```df_titanic_m.groupby('Survived').count()['Age'] / df_titanic.count()['Survived'] * 100``` #61.616162
# 

# In[292]:


# Gráfico de Distribuição de sobreviventes por idade (Histograma)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(df_titanic_s['Age'], alpha=0.5, label='Sobreviventes')
ax.hist(df_titanic_m['Age'], alpha=0.5, label='Mortos')
ax.set_title('Distribuição de sobreviventes e Mortos por Faixa Etária')
ax.set_xlabel('Idade')
ax.set_ylabel('Total')
ax.legend(loc='upper right')
plt.show()

# Sobreviventes
plt.hist(df_titanic_s['Age'], bins=12)
plt.title('Sobreviventes')
plt.show()

# Mortos
plt.hist(df_titanic_m['Age'], bins=12)
plt.title('Mortos')
plt.show()


# Observa-se que a grande maioria de mortos (61.1% contra 38.4% de sobreviventes), está dentro da faixa etária média (**28.5** anos) entre 20 e 30 anos. Um pequeno grupo de passageiros entre pouco mais de 70 a 80 anos sobreviveram.

# In[293]:


# Gráfico de Distribuição de sobreviventes homens x mulheres por Idade
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(df_titanic_s_h['Age'], alpha=0.5, label='Homens')
ax.hist(df_titanic_s_m['Age'], alpha=0.5, label='Mulheres')
ax.set_title('Distribuição de sobreviventes Homens e Mulheres por Faixa Etária')
ax.set_xlabel('Idade')
ax.set_ylabel('Total')
ax.legend(loc='upper right')
plt.show()


# As crianças entre 0 a pouco mais de 10 anos (meninos e meninas) tiveram praticamente o mesmo número de sobreviventes.
# 
# O Total de sobreviventes mulheres com pouco mais de 10 anos até a faixa dos 60 foi bem superior ao total de homens sobreviventes, tendo como a faixa dos 30 anos (média de **28.9** anos para as mulheres) aquela teve o maior número de sobreviventes. Alguns poucos passageiros homens na faixa dos 70-80 anos sobreviveram.
# 
# **Concluímos aqui que realmente mulheres e crianças salvaram-se primeiro.**

# ### Correlação de Mortos entre Classe e Faixa Etária

# In[312]:


# Gráfico de correlação de morte entre Classe e Faixa Etária
plt.scatter(df_titanic_m['Pclass'], df_titanic_m['Age'], alpha=0.3, c=df_titanic_m['Age'], s=df_titanic_m['Age'])
plt.title('Correlação de Mortes entre Classe e Faixa Etária')
plt.xticks([1,2,3], ['1.ª','2.ª','3.ª'])
plt.xlabel('Classe')
plt.ylabel('Idade')
plt.show()


# ## Relações envolvendo o Valor da Tarifa (Fare)

# In[303]:


df_titanic['Fare'].mean()


# In[302]:


# Valor médio da tarifa
df_titanic.groupby(['Pclass']).mean()['Fare']


# In[304]:


# Relação entre a idade e o valor pago da Tarifa
plt.scatter(df_titanic['Age'], df_titanic['Fare'], alpha=0.5, c=df_titanic['Fare'], s=df_titanic['Fare'])
plt.title('Relação entre Faixa Etária e Tarifa')
plt.show()


# No gráfico de dispersão acima, observa-se uma relação entre idade e tarifa, indicando que as passagens mais caras foram adquiridas pelas faixas de idades entre 30-40 anos (cor amarela), e uma faixa de 20-50 com valores acima de \$200 (cor verde), porém a grande maioria dos passageiros adquiriu as tafifas mais baratas (abaixo de \$100), indicando que além das classes mais baixas pagaram também, a 2.ª e a 3.ª classe, dentro da média de valores (\$32).

# ### Valor total da Tarifa agrupado por Classe, Local de Embarque (porto) e Gênero

# In[307]:


# Agrupamento por Classe, Porto e Gênero (contagem de passageiros)
df_titanic.groupby(by=['Pclass', 'Embarked', 'Sex'])['Name'].count()


# In[305]:


# Agrupamento por Classe, Porto e Gênero (soma das tarifas)
df_titanic['Embarked'] = df_titanic['Embarked'].replace(['C','Q','S'],['Cherbourg','Queenstown','Southampton'])
df_titanic.groupby(by=['Pclass', 'Embarked', 'Sex'])['Fare'].sum()


# #### Embarque (porto) x Classe
# - Predominância de embarque nos portos por classe:
#   - 1.ª classe predominatemente de passageiros que embarcaram em "Cherbourg" e "Southampton";
#   - 2.ª classe embarques em "Southampton" predominantemente;
#   - 3.ª classe embarques em "Southampton" e "Queenstown".

# ### Relação entre sobrevivência e tarifa não paga (gratuidade)
# Aqueles que não pagaram tarifa sobreviveram? Nome dos sobreviventes?

# In[326]:


df_titanic[df_titanic['Fare'] == 0]['Name'].tolist()


# In[308]:


# Listagem dos não pagantes (tarifa = 0)
df_titanic[df_titanic['Fare'] == 0].groupby(by=['Name', 'Age', 'Sex', 'Survived', 'Pclass'])['Fare'].sum()


# In[318]:


# Media de idade dos passageiros não pagantes (tarifa = 0)
df_titanic[df_titanic['Fare'] == 0].mean()['Age']


# In[58]:


# Preparação dos dados de passageiros que não tiveram a tarifa cobrada
df_titanic_zerofare = df_titanic[df_titanic['Fare'] == 0]

# Nome dos passageiros que não tiveram a tarifa cobrada e sobreviveram
df_titanic_zerofare[df_titanic_zerofare['Survived'] == 1]['Name']


# Nota: Mr. William Henry Tornquist foi o único sobrevivente que não pagou tarifa.

# ### Curiosidade: Outliers são os mais idosos

# In[60]:


# Age Boxplot
df_titanic_s['Age'].plot(kind='box');


# In[317]:


# Sobrevivente Outlier
df_titanic_s[df_titanic_s['Age'] >= 55]


# In[182]:


# Sobrevivente Outlier
df_titanic[df_titanic['Age'] >= 80]


# ***
# # Answering questions
# 
# ## Respondendo aos questionamentos de acordo com as conclusões tiradas
# 
# #### Dentre os questionamentos propostos com este conjunto de dados, estas forma as respostas através das análises realizadas:
# 
# 1 - Quais foram os fatores que fizeram com que algumas pessoas mais propensas a sobreviver?    
# **R: Gênero e Classe**
# 
# 2 - Será que a classe mais alta teve mais chances de sobrevivência que a mais baixa?  
# **R: Sim, a 1.ª classe teve melhores chances de sobrevovência **
# 
# 3 - A proporção de sobrevivência da primeira classe foi superior em relação à segunda e a terceira classe?  
# **R: Sim **
# 
# - Sobreviventes na 1.ª Classe = 136 (39.7%)
# - Sobreviventes na 2.ª Classe = 87 (25.4%)
# - Sobreviventes na 3.ª Classe = 119 (34.8%)
# 
# 4 - Há correlação direta entre os que sobreviveram e seus gêneros?  
# **R:  Sim, mulheres tiveram os melhores índices de sobrevivência**
# 
# 5 - Qual a proporção de sobreviventes e mortos por gênero?  
# **R: Homens: Sobreviveram 19% (109) e Morreram 81% (468) | Mulheres: Sobreviveram 74% (233) e Morreram 26% (81) **
# 
# 6 - Mulheres e crianças salvaram-se primeiro?  
# **R: Sim, as análises quantitativas por idade e gênero apontaram que a distribuição de sobreviventes foi bastante significativa entre 0 e pouco mais de 10 anos (entre meninos e meninas) e entre as mulheres **
# 
# 7 - Há correlação direta entre o valor pago na passagem e idade?  
# **R: Sim, as passagens mais caras foram adquiridas pelas faixas de idades entre 30-40 anos (cor amarela), e uma faixa de 20-50. Houveram gratuidades para idades com média de 32 anos **
# 
# 8 - Quem são os passageiros que não tiveram a tarifa cobrada? Eles sobreviveram?
# **R: ** 'Leonard, Mr. Lionel', 'Harrison, Mr. William', 'Tornquist, Mr. William Henry', 'Parkes, Mr. Francis "Frank"',
#  'Johnson, Mr. William Cahoone Jr', 'Cunningham, Mr. Alfred Fleming', 'Campbell, Mr. William', 'Frost, Mr. Anthony Wood "Archie"', 'Johnson, Mr. Alfred', 'Parr, Mr. William Henry Marsh', 'Watson, Mr. Ennis Hastings', 'Knight, Mr. Robert J', 'Andrews, Mr. Thomas Jr', 'Fry, Mr. Richard', 'Reuchlin, Jonkheer. John George'. ** Mr. William Henry Tornquist foi o único sobrevivente que não pagou tarifa. **

# ## Quais conclusões podemos tirar no naufrágio do Titanic?

# Uma das razões pelas quais o naufrágio causou tal perda de vidas foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Embora houvesse algum elemento de sorte envolvido na sobrevivência do naufrágio, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta.

# ### Referências
# 
# Stackoverflow [https://stackoverflow.com/]  
# Matplotlib [https://matplotlib.org/]  
# Pandas [https://pandas.pydata.org/]  
# 
# ##### Links
# 
# https://www.kaggle.com/c/titanic/data#  
# https://paulovasconcellos.com.br/o-que-o-naufr%C3%A1gio-do-titanic-nos-ensina-at%C3%A9-hoje-data-science-project-2fea8ff1c9b5  
# https://github.com/paulozip/naufragio-titanic  
# https://github.com/pandas-dev/pandas/issues/10611  
# https://www.youtube.com/watch?v=LKESQMPe7nI  
# https://www.youtube.com/watch?v=3eTSVGY_fIE  
# http://felipegalvao.com.br/blog/2016/03/08/visualizacao-de-dados-com-python-matplotlib/  
# https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html  
# https://www.kaggle.com/c/titanic  
# https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet  
# 
# ### Bibliografia
# Data Science do Zero - Grus - O'Reilly  
# Python para Análise de Dados - McKinney - O'Reilly | novatec

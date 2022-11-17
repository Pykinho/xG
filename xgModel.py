import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FCPython
import statsmodels.api as sm
import statsmodels.formula.api as smf

#combine the data
England = pd.read_json('./events/events_England.json')
shotOnly = England.eventName == 'Shot'
shotEngland = England[shotOnly]
European = pd.read_json('./events/events_European_Championship.json')
shotOnly = European.eventName == 'Shot'
shotEuropean = European[shotOnly]
France = pd.read_json('./events/events_France.json')
shotOnly = France.eventName == 'Shot'
shotFrance = France[shotOnly]
Germany = pd.read_json('./events/events_Germany.json')
shotOnly = Germany.eventName == 'Shot'
shotGermany = Germany[shotOnly]
Italy = pd.read_json('./events/events_Italy.json')
shotOnly = Italy.eventName == 'Shot'
shotItaly = Italy[shotOnly]
Spain = pd.read_json('./events/events_Spain.json')
shotOnly = Spain.eventName == 'Shot'
shotSpain = Spain[shotOnly]
WorldCup = pd.read_json('./events/events_World_Cup.json')
shotOnly = WorldCup.eventName == 'Shot'
shotWorldCup = WorldCup[shotOnly]
shots = pd.concat([shotEngland,shotEuropean,shotFrance,shotGermany,shotSpain,shotItaly,shotWorldCup])

shots_model = pd.DataFrame(columns=['Goal','X','Y'])
headers_model = pd.DataFrame(columns=['Goal','X','Y'])

for i,shot in shots.iterrows():
    header = 0
    for tag in shot['tags']:
        if tag['id'] == 403:
            header = 1

#open play shots only, no headers:
    if not header:
        shots_model.at[i,'X'] = 100 - shot['positions'][0]['x']
        shots_model.at[i,'Y'] = shot['positions'][0]['y']
        shots_model.at[i,'C'] = abs(shot['positions'][0]['y']-50)

    #distance
        x = shots_model.at[i,'X'] * 105/100
        y = shots_model.at[i,'C'] * 65/100
        shots_model.at[i,'Distance'] = np.sqrt(x**2 + y**2)

        a = np.arctan(7.32*x/(x**2+y**2-(7.32/2)**2)) #skąd to 7.32?
        if a<0:
            a += np.pi

        shots_model.at[i,'Angle'] = a

        #Goal
        shots_model.at[i,'Goal'] = 0

        for tag in shot['tags']:
            if tag['id'] == 101:
                shots_model.at[i,'Goal'] = 1
                
                
#headers:
    else:
        headers_model.at[i,'X'] = 100 - shot['positions'][0]['x']
        headers_model.at[i,'Y'] = shot['positions'][0]['y']
        headers_model.at[i,'C'] = abs(shot['positions'][0]['y']-50)

    #distance
        x = headers_model.at[i,'X'] * 105/100
        y = headers_model.at[i,'C'] * 65/100
        headers_model.at[i,'Distance'] = np.sqrt(x**2 + y**2)

        a = np.arctan(7.32*x/(x**2+y**2-(7.32/2)**2)) #skąd to 7.32?
        if a<0:
            a += np.pi

        headers_model.at[i,'Angle'] = a

        #Goal
        headers_model.at[i,'Goal'] = 0

        for tag in shot['tags']:
            if tag['id'] == 101:
                headers_model.at[i,'Goal'] = 1


model_variables = ['Angle','Distance']
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]



#Fit the model
test_model = smf.glm(formula="Goal ~ " + model, data=shots_model, 
                           family=sm.families.Binomial()).fit()
print(test_model.summary())        
b=test_model.params

#Fit the header model
test_headermodel = smf.glm(formula="Goal ~ " + model, data=headers_model, 
                           family=sm.families.Binomial()).fit()
print(test_headermodel.summary())        
bh=test_headermodel.params


#export the coefficients
with open("coefficients.txt",'w') as file:
    file.write('[official]' + '\n')
    file.write('intercept = ' + str(b[0]) + '\n')
    file.write('angle = ' + str(b[1]) + '\n')
    file.write('distance = ' + str(b[2]) + '\n')
    file.write('intercept_h = ' + str(bh[0]) + '\n')
    file.write('angle_h = ' + str(bh[1]) + '\n')
    file.write('distance_h = ' + str(bh[2]) + '\n')


#Return xG value for more general model
def calculate_xG(sh):    
   bsum=b[0]
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*sh[v]
   xG = 1/(1+np.exp(bsum)) 
   return xG   


#Return xG value for more general header model
def calculate_header_xG(sh):    
   bsum=bh[0]
   for i,v in enumerate(model_variables):
       bsum=bsum+bh[i+1]*sh[v]
   xG = 1/(1+np.exp(bsum)) 
   return xG 

#add xG to data frames
xG=shots_model.apply(calculate_xG, axis=1) 
shots_model = shots_model.assign(xG=xG)

xG=headers_model.apply(calculate_header_xG, axis=1) 
headers_model = headers_model.assign(xG=xG)
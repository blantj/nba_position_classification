#Import libraries
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load

#
st.title('NCAA Basketball Player Position Classifier')
st.markdown('Select a 2018-19 NCAA player to see how their position is classified')

#
df = pd.read_csv('Data/ncaa_dataset.csv')
players = df['Name-Team'].values

#
label = 'Enter 2018-19 NCAA player name'
player = st.selectbox(label, players)

#
player_datapoint = df[df['Name-Team']==player]
x = player_datapoint.drop(columns=['Name-Team', 'Position', 'G', 'GS', 'MP'])
y = player_datapoint['Position'].values[0]

#
ss = load(open('Pickles/standard_scaler.pickle', 'rb'))
x = ss.transform(x)

#
svm = load(open('Pickles/svm.pickle', 'rb'))
y_pred = svm.predict(x)[0]

#
columns = ['Player Name', 'Team', 'Predicted Class', 'Actual Class']
output = pd.DataFrame(columns=columns)
output.loc[0] = [player.split(' / ')[0], player.split(' / ')[-1], y_pred, y]
output_dict = {0: 'Guard', 1: 'Forward'}
output.replace(output_dict, inplace=True)
output.index = [' ']


#
st.table(output)
#1 задание выполнял вместе с Кутиловым Богданом, проверил Кашин Матвей
from sympy import *
k,T,C,L = symbols('k T C L')
#1-й способ 100000
# 
Am_lst=[]
C_ost_lst=[]
print ('индивидуальное задание 4 вариант')
C_ost = 40000
Am_lst=[]
C_ost_lst=[]
for i in range (10):
  Am=(C-L)/T  
  C_ost -= Am.subs({C:40000, T:10, L:0})
  Am_lst.append(round(Am.subs({C:40000, T:10, L:0}),2))
  C_ost_lst.append(round(C_ost,2))
print('Am_lst: ', Am_lst)
print('C_ost_lst: ',C_ost_lst)
#2-й способ
Aj=0
C_ost=40000
Am_lst_2=[]
C_ost_lst_2=[]
for i in range (10):
  Am= k*1/T*(C-Aj)
  C_ost-= Am.subs({C:40000, T:10, k:2})
  Am_lst_2.append(round(Am.subs({C:40000, T:10, k:2}),2))
  Aj+=Am.subs({C:40000, T:10, k:2})
  C_ost_lst_2.append(round(C_ost,2))
print ('Am_lst_2: ', Am_lst_2)
print ('C_ost_lst_2: ',C_ost_lst_2)
#Представление в таблице
import pandas as pd
Y= range(1,11)
table1=list(zip(Y,Am_lst,C_ost_lst))
table2=list(zip(Y,Am_lst_2,C_ost_lst_2))
tframe=pd.DataFrame(table1,columns=['Y','Am_lst','C_ost_lst'])
tframe2=pd.DataFrame(table2,columns=['Y','Am_lst_2','C_ost_lst_2'])
print(tframe)
print(tframe2)

#контейнер визуализации
import numpy as np 
import matplotlib.pyplot as plt# Что это означает? подключение библиотеки для графиков и обозначает ссылку на нее plt
plt.figure()
plt.plot(tframe['Y'],tframe['C_ost_lst'],label='Am_lst')
plt.savefig('chart7.png')
plt.figure() 
plt.plot(tframe2['Y'],tframe2['C_ost_lst_2'],label='Am_lst_2')
plt.savefig('chart8.png')
#круговая диаграмма по 1 способу
vals=Am_lst
labels=[str(x) for x in range(1,11)]# Что это означает? создание списка из 6 элементов
explode =(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
fig,ax=plt.subplots()
ax.pie(vals,explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,wedgeprops={'lw':1,'ls':'--','edgecolor': "k"},rotatelabels=True)
ax.axis("equal")
plt.savefig('chart9.png')
#круговая диаграмма по 2 способу
vals=Am_lst_2
labels=[str(x) for x in range(1,11)]
explode =(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
fig,ax=plt.subplots() # Что это означает?
ax.pie(vals,explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,wedgeprops={'lw':1,'ls':'--','edgecolor': "k"},rotatelabels=True)
ax.axis("equal")
plt.savefig('chart10.png')

table1 = list(zip(Y, Am_lst))
table2 = list(zip(Y, Am_lst_2))
tframe = pd.DataFrame(table1, columns=['Y', 'Am_lst'])
tframe2 = pd.DataFrame(table2, columns=['Y', 'Am_lst_2']) # Что это означает? разбивка данных таблицы на оси и их подписи
plt.figure()
plt.bar(tframe['Y'], tframe['Am_lst'])
plt.savefig('chart11.png')
plt.close()
plt.figure()
plt.bar(tframe2['Y'], tframe2['Am_lst_2'])
plt.savefig('chart12.png')
plt.close()

#нет исходных данных и указаний варианта , 4,5/5
# на вопросы ответил Кашин Матвей
# 5 задание с SHell
# 6 задание. Подвязан аккаунт на гитхаб и подгружены файлы на гидхам
# 7 задание. Внесены изменения в соответсвии с новыми исходными данными: сумма 40000, срок 10 лет
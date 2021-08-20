import pandas
data=pandas.read_excel('tabl.xlsx',index_col=0)
data.head()
data.salary.describe()
data.salary.hist()
data.city.value_counts().plot(kind='bar')
final=pandas.get_dummies(data,columns=['city','vacation_preference','transport_preference'])
x=final.drop('target',axis=1)
y=final.target
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(x,y)
print(forest.get_params())
inp=x.head(1)
print({col:[0] for col in x.columns })
example={'age': [20],
 'city_Екатеринбург': [0],
 'city_Киев': [0],
 'city_Краснодар': [0],
 'city_Минск': [0],
 'city_Москва': [1],
 'city_Новосибирск': [0],
 'city_Омск': [0],
 'city_Петербург': [0],
 'city_Томск': [0],
 'city_Хабаровск': [0],
 'city_Ярославль': [0],
 'family_members': [0],
 'salary': [3001000],
 'transport_preference_Автомобиль': [0],
 'transport_preference_Космический корабль': [1],
 'transport_preference_Морской транспорт': [0],
 'transport_preference_Поезд': [0],
 'transport_preference_Самолет': [0],
 'vacation_preference_Архитектура': [1],
 'vacation_preference_Ночные клубы': [0],
 'vacation_preference_Пляжный отдых': [0],
 'vacation_preference_Шоппинг': [0]}
ex=pandas.DataFrame(example)
print(forest.predict(ex))
print(forest.predict(ex))



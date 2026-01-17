
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df_a = pd.read_csv('./data/A.csv')  
df_b = pd.read_csv('./data/B.csv')  
df_c = pd.read_csv('./data/C.csv')  
df_d = pd.read_csv('./data/D.csv')  
df_e = pd.read_csv('./data/E.csv')  


data = pd.concat([df_a, df_b, df_c, df_d, df_e], ignore_index=True)

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Model accuracy on test data: {accuracy*100:.2f}%")

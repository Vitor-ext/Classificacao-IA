# classificacao.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Carregando o conjunto de dados de câncer de mama
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target variable (0 - benigno, 1 - maligno)

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Criando e treinando um modelo de regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Aplicando a transformação do scaler nos dados de teste
X_test_scaled = scaler.transform(X_test)

# Realizando previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Exibindo os resultados
print("Accuracy:", accuracy, "\n")

print("\nConfusion Matrix:\n", conf_matrix)
print("             Predito Negativo  |  Predito Positivo \nReal Negativo   TN                     FP\nReal Positivo   FN                     TP")

print("\nClassification Report:\n", class_report)

import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle


def load_data(path='Dataset'):
    X, Y = [], []
    for root, dirs, files in os.walk(path):
        label = os.path.basename(root)
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64)) 
                img = img.astype('float32') / 255 
                X.append(img)
                Y.append(0 if label == 'normal' else 1)  
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], -1)  
    return X, Y


def train_models():
    X, Y = load_data()  
    num_components = min(X.shape[0], X.shape[1]) 
    pca = PCA(n_components=num_components)  
    X = pca.fit_transform(X)  

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  

    
    svm_model = svm.SVC()  
    svm_model.fit(X_train, y_train)  
    svm_acc = accuracy_score(y_test, svm_model.predict(X_test)) * 100  

    kmeans = KMeans(n_clusters=2, random_state=0)  
    kmeans.fit(X_train)  
    kmeans_acc = accuracy_score(y_test, kmeans.predict(X_test)) * 100  

    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)

    return svm_model, pca, svm_acc, kmeans_acc


def predict_cancer(img, svm_model, pca):
    img = cv2.resize(img, (64, 64))  
    img = img.astype('float32') / 255  
    img = img.reshape(1, -1)  
    img_pca = pca.transform(img)  

    
    prediction = svm_model.predict(img_pca)  

    
    return "Abnormal" if prediction[0] == 1 else "Normal"


if __name__ == '__main__':
    svm_model, pca, svm_acc, kmeans_acc = train_models()
    print(f"SVM Accuracy: {svm_acc:.2f}%")
    print(f"KMeans Accuracy: {kmeans_acc:.2f}%")

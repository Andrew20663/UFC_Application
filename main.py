#Load all necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import shap
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import xgboost 
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import auc
from explainerdashboard import ClassifierExplainer
from sklearn.inspection import plot_partial_dependence
from bayes_opt import BayesianOptimization
import streamlit as st

data = st.beta_container()
red_fighter, blue_fighter = st.beta_columns(2)
exploratory_data_analysis = st.beta_container()
machine_learning_models = st.beta_container()
cross_validation_results = st.beta_container()
hyperparameter_tuning = st.beta_container()
final_model = st.beta_container()

with data:
    st.title('UFC Streamlit Application')
    df = pd.read_csv('ufc-master.csv')
    st.subheader('UFC Dataset')
    st.write(df.head())
    st.subheader('UFC Summary Statistics')
    st.write(df.describe().T)

with red_fighter:
    st.subheader('Red Fighter Stats')
    choice = st.selectbox('Choose Red Fighter', options = df.loc[:, 'R_fighter'])
    st.write(df.loc[df['R_fighter'] == choice, ['R_fighter', 'R_odds', 'R_current_win_streak', 'B_fighter', 'B_odds', 'B_current_win_streak']])

with blue_fighter:
    st.subheader('Blue Fighter Stats')
    choice = st.selectbox('Choose Blue Fighter', options = df.loc[:, 'B_fighter'])
    st.write(df.loc[df['B_fighter'] == choice, ['R_fighter', 'R_odds', 'R_current_win_streak', 'B_fighter', 'B_odds', 'B_current_win_streak']])

with exploratory_data_analysis:
    st.subheader('Boxplot of Red and Blue Fighter Odds')
    #Instantiate Figure class
    fig = go.Figure()
    #Adjust components of plot for red fighter
    fig.add_trace(go.Box(y = df['R_odds'], boxpoints = 'all', name = 'Red Fighter', line = dict(color = 'red')))
    #Adjust componet of plot for blue fighter
    fig.add_trace(go.Box(y = df['B_odds'], boxpoints = 'all', name = 'Blue Fighter', line = dict(color = 'blue')))

    #Adjust layout overall
    fig.update_layout(

    title = 'Red Fighter and Blue Fighter Odds Boxplot',
    paper_bgcolor = 'white',
    xaxis_title = 'Fighter Color',
    yaxis_title = 'Odds')
    st.plotly_chart(fig)

    st.subheader('Number of Fighters in Each Weight Class')
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize = (10,10))
    sns.countplot(data = df, y = 'weight_class', order = df['weight_class'].value_counts().index, ax = ax)
    ax.set_title('Number of fighters in Each Weight Class', fontsize = 20)
    ax.set_xlabel('Number of Fighters', fontsize = 10)
    ax.set_ylabel('Weight Class', fontsize = 10)
    for i in ax.patches:
        ax.annotate(i.get_width(), (i.get_x()+i.get_width(), i.get_y()+i.get_height()))
    st.pyplot(fig)

    st.subheader('10 Most Popular Finishes')
    fig, ax = plt.subplots(figsize = (10,10))
    sns.countplot(data = df, y = 'finish_details', order = df['finish_details'].value_counts()[0:10].index)
    ax.set_title('10 Most Popular Finishes', fontsize = 20)
    ax.set_xlabel('Number of Finishes', fontsize = 10)
    ax.set_ylabel('Finish Type')
    for i in ax.patches:
        ax.annotate(i.get_width(), (i.get_x() + i.get_width(), i.get_y() + i.get_height()))
    st.pyplot(fig)

    for i in df.columns:
        na = df[i].isna().sum() / len(df)
        if na >= 0.7:
            df.drop(i, axis = 1, inplace = True)
    
    df = df.dropna(subset = ['R_kd_bout', 'B_kd_bout', 'R_sig_str_landed_bout',
       'B_sig_str_landed_bout', 'R_sig_str_attempted_bout',
       'B_sig_str_attempted_bout', 'R_sig_str_pct_bout', 'B_sig_str_pct_bout',
       'R_tot_str_landed_bout', 'B_tot_str_landed_bout',
       'R_tot_str_attempted_bout', 'B_tot_str_attempted_bout',
       'R_td_landed_bout', 'B_td_landed_bout', 'R_td_attempted_bout',
       'B_td_attempted_bout', 'R_td_pct_bout', 'B_td_pct_bout',
       'R_sub_attempts_bout', 'B_sub_attempts_bout', 'R_pass_bout',
       'B_pass_bout', 'R_rev_bout', 'B_rev_bout', 'finish_details'])
    
    median = []
    for i in ['B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed','B_avg_TD_pct', 'R_avg_SIG_STR_landed', 'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_avg_TD_pct']:
        median = df[i].median()
        df[i] = df[i].fillna(median)
    col = []
    for i in df.columns:
        if df[i].dtypes == 'O':
            col.append(i)
    col.append('date')
    #one hot encode categorical features with pandas get_dummies
    ufc_dummies = pd.get_dummies(df.loc[:, ['R_fighter', 'B_fighter', 'location', 'country', 'Winner',
    'weight_class', 'gender', 'B_Stance', 'R_Stance', 'better_rank', 'finish', 'finish_details', 'finish_round_time', 'date']])
    df = pd.concat([df, ufc_dummies], axis = 1)
#drop dataframe columns with categorical string 
    df = df.drop(columns = ['location', 'country', 'Winner', 'R_fighter', 'B_fighter', 'Winner',
    'weight_class', 'gender', 'B_Stance', 'R_Stance', 'better_rank', 'finish', 'finish_details', 'finish_round_time', 'date'])
    #drop winner_blue column
    df = df.drop('Winner_Blue', axis = 1)
    #keep winner_red to distinguish red winner and rename to winner
    df = df.rename({'Winner_Red' : 'Winner'}, axis = 1)

with machine_learning_models:
    X = df.drop('Winner', axis = 1)
    y = df['Winner']
    correlation_matrix = X.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k = 1).astype(np.bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.8)]
    X = X.drop(drop, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)


    st.subheader('Choose model you would like to see metrics for')
    choice = st.selectbox('Model', options = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), GradientBoostingClassifier(),
    RandomForestClassifier(), SVC(), MLPClassifier(), SGDClassifier()], index = 3)
    m = choice 
    m.fit(X_train, y_train)
    predictions = m.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    st.write('model:', choice)
    st.write('accuracy:', accuracy)
    st.write('precision:', precision)
    st.write('recall:', recall)
    st.write('f1:', f1)
    st.write('auc', roc_auc)

    st.subheader('Precision and Recall with Threshold Curve')
    m = choice
    m.fit(X_train, y_train)
    y_score = choice.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    fig = plt.figure(figsize=(10,10))
    plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
    plt.plot(thresholds, precision[:-1], 'r-', label = 'Precision')
    plt.xlabel('Threshold', fontsize = 15)
    plt.title('Precision vs. Recall with Threshold', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Threshold', fontsize = 15)
    plt.legend(fontsize = 20)
    st.pyplot(fig)

    fig = plt.figure(figsize = (10,10))
    plt.plot(precision, recall)
    plt.title('Precision vs. Recall', fontsize = 20)
    plt.xlabel('Recall', fontsize = 15)
    plt.ylabel('Precision', fontsize = 15)
    st.pyplot(fig)
    

    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    y_score = gb.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    #histogram of scores compared to true labels
    fig_hist = px.histogram(x = y_score, color = y_test, nbins = 50,
    labels = dict(color = 'True Labels', x = 'score'), title = 'Histogram of True Labels', opacity = 0.7,
    marginal = 'box')
    st.plotly_chart(fig_hist)

    df_tp = pd.DataFrame({
    'false positive rate' : fpr,
    'true positive rate' : tpr
}, index = thresholds)
    df_tp.index.name = 'thresholds'
    df_tp.columns.name = 'rate'

    fig_thresh = px.line(df_tp, title = 'True Positive Rate (TPR) and and False Positive Rate (FPR) at every threshold')

    fig_thresh.update_yaxes(scaleanchor = 'x', scaleratio = 1)
    fig_thresh.update_xaxes(range = [0,1], constrain = 'domain')
    fig_thresh.update_yaxes(scaleanchor = 'x', scaleratio = 1)
    fig_thresh.update_xaxes(range = [0,1], constrain = 'domain')
    st.plotly_chart(fig_thresh)
    
    fig = px.area(x = fpr, y = tpr, 
    title = 'ROC Curve of Gradient Boosting Tree Classifier',
    labels = dict(x = 'False Positive Rate', y = 'True Positive Rate'))

    fig.add_shape(
        type = 'line', line = dict(dash = 'dash'),
        x0 = 0, x1 = 1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor = 'x', scaleratio = 1)
    fig.update_xaxes(range = [0,1], constrain = 'domain')
    st.plotly_chart(fig)

with cross_validation_results:
    
    st.subheader('Cross Validation Results')
    st.text('Choose model to see cross validation results')
    cv = st.selectbox('Cross Validation Model', options = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), GradientBoostingClassifier(),
    RandomForestClassifier(), SVC(), MLPClassifier(), SGDClassifier()], index = 3)
    kf = KFold(n_splits = 10, random_state = 123, shuffle = True)
    cv_score_accuracy = cross_val_score(cv, X_train, y_train, cv = kf, scoring = 'accuracy')
    cv_score_precision = cross_val_score(cv, X_train, y_train, cv = kf, scoring = 'precision')
    cv_score_recall = cross_val_score(cv, X_train, y_train, cv = kf, scoring = 'recall')
    cv_score_f1 = cross_val_score(cv, X_train, y_train, cv = kf, scoring = 'f1')
    cv_score_roc_auc = cross_val_score(cv, X_train, y_train, cv = kf, scoring = 'roc_auc')
    mean_accuracy = np.mean(cv_score_accuracy)
    mean_precision = np.mean(cv_score_precision)
    mean_recall = np.mean(cv_score_recall)
    mean_f1 = np.mean(cv_score_f1)
    mean_roc_auc = np.mean(cv_score_roc_auc)
    
    st.write('model:', cv)
    st.write('accuracy:', mean_accuracy)
    st.write('precision:', mean_precision)
    st.write('recall:', mean_recall)
    st.write('f1:', mean_f1)
    st.write('auc', mean_roc_auc)   


with hyperparameter_tuning:
    st.subheader('Hyperparameter Tuning')
    @st.cache(allow_output_mutation=True)
    def hyperparameters(cv = 10):
        param_grid = [
        {'loss' : ['deviance', 'exponential']},
        {'learning_rate' : [.0001, .001, .01, .1, 1, 10, 100, 1000]},
        {'n_estimators' : [1, 10, 100, 1000, 10000]},
        {'subsample' : [.0001, .001, .01, .1, 1]},
        {'criterion' : ['friedman_mse', 'mse', 'mae']},
        {'min_samples_split' : [.0001, .001, .01, 1, 10, 100]},
        {'min_samples_leaf' : [.0001, .001, .01, 1, 10, 100]},
        {'min_weight_fraction_leaf' : [.0001, .001, .01, .1, 1, 10, 100]},
        {'max_depth' : [1,3,6,9,12,15,18,21,24,27,30]},
        {'min_impurity_decrease' : [0, .0001, .001, .01, .1, 1, 10, 100]},
        {'min_impurity_split' : [.0001, .001, .01, .1, 1, 10]},
        {'max_features' : ['auto', 'sqrt', 'log2']}
        ]

        gb = GradientBoostingClassifier()
        randomized_search = RandomizedSearchCV(gb, param_distributions = param_grid, cv = 10, scoring = 'accuracy', return_train_score = True)
        results = randomized_search.fit(X_train, y_train)
        return results 

    #choice = st.slider('Choose your learning rate', value = [.001, .01, .1, 1.])
    randomized_search_cv = hyperparameters(cv = 10)
    st.write('Randomized Search Best Parameters')
    st.write(randomized_search_cv.best_params_)
    st.write('Randomized Search Best Estimator')
    st.write(randomized_search_cv.best_estimator_)
    
    with final_model:
        st.subheader('Final Accuracy and Metrics')
        final_model = randomized_search_cv.best_estimator_
        final_predictions = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, final_predictions)
        final_precision = precision_score(y_test, final_predictions, average = 'weighted')
        final_recall = recall_score(y_test, final_predictions, average = 'weighted')
        final_f1 = f1_score(y_test, final_predictions, average = 'weighted')
        final_roc_auc = roc_auc_score(y_test, final_predictions)
        st.write(('final accuracy:', final_accuracy), '\n',('final precision:', final_precision), '\n', ('final recall:', final_recall), '\n', ('final f1:', final_f1), '\n', ('final auc:', final_roc_auc))

        st.subheader('Precision vs. Recall with Threshold Final Model')
        y_final_score = final_model.decision_function(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_final_score)
        fig = plt.figure(figsize=(10,10))
        plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
        plt.plot(thresholds, precision[:-1], 'r-', label = 'Precision')
        plt.xlabel('Threshold', fontsize = 15)
        plt.title('Precision vs. Recall with Threshold Final Model', fontsize = 20)
        plt.legend(fontsize = 15)
        st.pyplot(fig)

        st.subheader('Precision vs. Recall Final Model')
        fig = plt.figure(figsize = (10,10))
        plt.plot(precision, recall)
        plt.title('Precision vs. Recall Final Model', fontsize = 20)
        plt.xlabel('Recall', fontsize = 15)
        plt.ylabel('Precision', fontsize = 15)
        st.pyplot(fig)

        st.subheader('Histogram of True Labels, True Positive Rate/False Positive Rate at Every Threshold and ROC Curve of Final Model')
        y_score_final = final_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score_final)

        #histogram of scores compared to true labels
        fig_hist = px.histogram( x = y_score_final, color = y_test, nbins = 50, 
        labels = dict(color = 'True Labels', x = 'score'), title = 'Histogram of True Labels with Scores Final Model', opacity = 0.7,
        marginal = 'box')
        st.plotly_chart(fig_hist)


        df_tp = pd.DataFrame({
            'false positive rate' : fpr,
            'true positive rate' : tpr
        }, index = thresholds)
        df_tp.index.name = 'thresholds'
        df_tp.columns.name = 'rate'

        fig_thresh = px.line(df_tp, title = 'TPR and FPR at every threshold Final Model',
        width = 700, height = 700)

        fig_thresh.update_yaxes(scaleanchor = 'x', scaleratio = 1)
        fig_thresh.update_xaxes(range = [0,1], constrain = 'domain')
        st.plotly_chart(fig_thresh)

        fig = px.area( x = fpr, y = tpr, 
        title = f'ROC Curve with AUC = {roc_auc_score(y_test,final_predictions):.4f}',
        labels = dict(x = 'False Positive Rate', y = 'True Positive Rate'),
        width = 700, height = 500)

        fig.add_shape(
            type = 'line', line = dict(dash = 'dash'),
            x0 = 0, x1 = 1, y0=0, y1=1
        )

        fig.update_yaxes(scaleanchor = 'x', scaleratio = 1)
        fig.update_xaxes(range = [0,1], constrain = 'domain')
        st.plotly_chart(fig)

        st.subheader('Confusion Matrix')
        st.write(confusion_matrix(y_test, final_predictions))

        st.subheader('Classification Report')
        st.write(classification_report(y_test, final_predictions))

        st.subheader('Feature Importance')
        feat_plot = plt.figure(figsize = (20,20))
        idx = np.argsort(np.abs(final_model.feature_importances_))[-20:][1::]
        feature_importances_sorted = np.abs(final_model.feature_importances_[idx])
        col_names = X_test.columns[idx]
        plt.barh(col_names, feature_importances_sorted)
        plt.title('Feature Importance of Gradient Boosted Model', fontsize = 20)
        plt.xlabel('Feature Importance', fontsize = 15)
        plt.ylabel('Features', fontsize = 15)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        st.pyplot(feat_plot)





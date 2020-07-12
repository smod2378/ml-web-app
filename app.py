'''
cd Desktop
mkdir- ML_DA
cd ML_DA
touch app.py
ls
streamlit run app.py
''' 

# Core Packages
import streamlit as st

# EDA (exploratory data analysis) packages
import pandas as pd
import numpy as np

# Data Visualization Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Machine Learning Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def main():
	html_page = """
	<div style="text-align: center;">
	<div>
		<h1>Machine Learning | Pedictive Analysis | Data Analytics</h1>
	</div>
	<div>
		<h2>Drag and Drop any dataset and predict or analyse</h2>
	</div>
	</div>
	"""
	st.markdown(html_page, unsafe_allow_html = True)

	activities = ["Exploratory Data Analysis", "Plot", "Model Building", "About"]
	choice = st.sidebar.selectbox("Select Activities", activities)
	if choice == "Exploratory Data Analysis":
		st.subheader("Exploratory Data Analysis")

		# Drag and Drop Feature
		data = st.file_uploader("Upload Dataset", type = ["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show Shape of the dataset"):
				st.write(df.shape)

			if st.checkbox("Show Columns of the dataset"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Select Columns to Show"):
				all_columns = df.columns.to_list() # doing this again so that we can use this functionality without depending on above
				selected_columns = st.multiselect("Select the Columns", all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Summary of the dataset"):
				st.write(df.describe())

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())


	elif choice == "Plot":
		st.subheader("Data Visualization")

		data = st.file_uploader("Upload Dataset", type = ["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

		if st.checkbox("Correlation Map with Seaborn"):
			st.write(sns.heatmap(df.corr(), annot = True))
			st.pyplot()

		if st.checkbox("Pie Chart"):
			all_columns = df.columns.to_list()
			columns_to_plot = st.selectbox("Select 1 Column", all_columns)
			pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()

		all_columns_names = df.columns.to_list()
		type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
		selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

		if st.button("Generate Plot"):
			st.success("Generating Customizable Plot of {} for {}" .format(type_of_plot, selected_columns_names))

			# Plots By Streamlit
			if type_of_plot == 'area':
				cust_data = df[selected_columns_names]
				st.area_chart(cust_data)
			elif type_of_plot == 'bar':
				cust_data = df[selected_columns_names]
				st.bar_chart(cust_data)
			elif type_of_plot == 'line':
				cust_data = df[selected_columns_names]
				st.line_chart(cust_data)

			# custom plot
			elif type_of_plot:
				cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
				st.write(cust_plot)
				st.pyplot()

	elif choice == "Model Building":
		st.subheader("Building Machine Learning Model")

		data = st.file_uploader("Upload Dataset", type = ["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			# Model Building
			X = df.iloc[:,0:-1] 
			Y = df.iloc[:,-1]
			seed = 7

			# model
			models = []
			models.append(("LR", LogisticRegression()))
			models.append(("LDA", LinearDiscriminantAnalysis()))
			models.append(("KNN", KNeighborsClassifier()))
			models.append(("CART", DecisionTreeClassifier()))
			models.append(("NB", GaussianNB()))
			models.append(("SVM", SVC()))

			# Evaluate each model one by one

			# List
			model_names= []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'

			for name,model in models:
				kfold = model_selection.KFold(n_splits=10, random_state = seed)
				cv_results = model_selection.cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())

				accuracy_result = {"model_name":name,"model_accuracy":cv_results.mean(),"Standard_deviation":cv_results.std()}
				all_models.append(accuracy_result)
			if st.checkbox("Metrics as Table"):
				st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std), columns=["Model Name", "Model Accuracy", "Standard Deviation"]))

			if st.checkbox("Metrics as JSON"):
				st.json(all_models)

	elif choice == "About":
		st.subheader("About Developer")

		html_page= """
		<div style = "background-color:tomato; padding:50px">
   		<p style="font-size:25px"> SHAIL MODI <br> B. Tech - Computer Engineering <br> &#9993; shail.modi1999@gmail.com <br> K.J.Somaiya College Of Engineering </p>
   		<p>&copy; shailmodi<p>
		</div>
		"""
		st.markdown(html_page, unsafe_allow_html = True)



if __name__=='__main__':
	main()

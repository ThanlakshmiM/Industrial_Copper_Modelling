import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle 

df=pd.read_csv(r"C:\Users\user\Desktop\python\Copper_Industrial\Copper_Modelling.csv")
df1=pd.read_csv(r"C:\Users\user\Desktop\python\Copper_Industrial\Copper_Modelling.csv")

eng = LabelEncoder()
eng.fit(df1[['status']])
df1['status'] = eng.transform(df1[['status']])
eng.fit(df1[['item type']])
df1['item type'] = eng.transform(df1[['item type']])
eng.fit(df1[['material_ref']])
df1['material_ref'] = eng.transform(df1[['material_ref']])

def Numeric(value,select):
    status=df["status"].unique()
    item_type=df["item type"].unique()
    matrial=df['material_ref'].unique()
    if select == "status" :
        for i in status:
            if i == value:
                status_map=dict(zip(status,df1["status"].unique()))
                return status_map[value]
    if select == "item_type" :
        for i in item_type:
            if i == value:
                item_map=dict(zip(item_type,df1["item type"].unique()))
                return item_map[value]
    if select == "matrial" :
        for i in matrial:
            if i == value:
                matrial_map=dict(zip(matrial,df1["material_ref"].unique()))
                return matrial_map[value]
            
def status(value):
    if value == 'Won':
        return 1
    else:
      if value == 'Lost':
          return 0

st.write("""

    Industrial Copper Modeling Application

""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])
with tab1:
    with st.form('my_form'):
            col1, col2, col3 = st.columns([5, 4, 5])
        
            col1.write(' ')
            status = col1.selectbox('Status',sorted(df["status"].unique()), key=1)
            item_type = col1.selectbox('Item Type',sorted(df["item type"].unique()), key=2)
            country =col1.selectbox('Country', sorted(df["country"].unique()), key=3)
            application = col1.selectbox('Application', sorted(df["application"].unique()), key=4)
            product_ref = col1.selectbox('Product Reference', sorted(df["product_ref"].unique()), key=5)
            matrial = col1.selectbox('Matrial Reference', df['material_ref'].unique(),key=6)
        
        
            col2.write(' ')
            item_year = col2.selectbox('item year',[2021, 2020],key=7)
            item_month = col2.selectbox('item month',sorted([ 4, 12,  3,  2,  1, 11, 10,  9,  8,  7]),key=8)
            item_day = col2.selectbox('item day',sorted([ 1,  2, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 19, 18, 17, 16, 15,
                                  14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3, 21, 20]),key=9)
            delivery_year = col2.selectbox('delivery year',[2021, 2022, 2020, 2019],key=10)
            delivery_month = col2.selectbox('delivery month',sorted([ 7,  4,  1,  3,  6,  5,  8,  9, 10, 11,  2, 12]),key=11)
            delivery_day = col2.selectbox('delivery day',[1],key=12)
            
         
            quantity_tons = col3.text_input(f'Enter Quantity Tons (Min:{df["quantity tons"].min()}, Max:{df["quantity tons"].max()})')
            thickness = col3.text_input(f'Enter thickness (Min:{df["thickness"].min()}, Max:{df["thickness"].max()})')
            width = col3.text_input(f'Enter width(Min:{df["width"].min()}, Max:{df["width"].max()})')
            customer = col3.text_input(f'customer ID (Min:{df["customer"].min()}, Max:{df["customer"].max()})')
            
            #input_data=np.array([[54.151139,30156308.0,28.0,7,5,10.0,2.00,1500.0,5384,1670798778,2021,4,1,2021,7,1]])
            input_data=np.array([[quantity_tons,customer,country,Numeric(status,"status"),Numeric(item_type,"item_type"),application,thickness,width,Numeric(matrial,"matrial"),product_ref,item_year,item_month,item_day,delivery_year,delivery_month,delivery_day]])
                  
            if st.form_submit_button(label='PREDICT SELLING PRICE'):
                                    
                    scaler_trans = pickle.load(open(r"C:\Users\user\Desktop\python\Copper_Industrial\Regression_scaler.pkl",'rb'))
                    result = pickle.load(open(r"C:\Users\user\Desktop\python\Copper_Industrial\DecisionTreeRegressor.pkl",'rb'))

                    scaled_new_data_point = scaler_trans.transform(input_data)
                    predicted_output = result.predict(scaled_new_data_point)
                    if predicted_output:
                        st.write(predicted_output[0])
                    else:
                        st.write("Enter correct format")       
with tab2:
    with st.form('my_form1'):
            col1, col2, col3 = st.columns([5, 4, 5])
        
            col1.write(' ')
           
            item_type = col1.selectbox('Item Type',sorted(df["item type"].unique()), key=13)
            country = col1.selectbox('Country', sorted(df["country"].unique()), key=14)
            application = col1.selectbox('Application', sorted(df["application"].unique()), key=15)
            product_ref = col1.selectbox('Product Reference', sorted(df["product_ref"].unique()), key=16)
            matrial = col1.selectbox('Matrial Reference', df['material_ref'].unique(),key=17)
        
            col2.write(' ')   
            item_year = col2.selectbox('item year',[2021, 2020],key=18)
            item_month = col2.selectbox('item month',sorted([ 4, 12,  3,  2,  1, 11, 10,  9,  8,  7]),key=19)
            item_day = col2.selectbox('item day',sorted([ 1,  2, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 19, 18, 17, 16, 15,
                                  14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3, 21, 20]),key=20)
            delivery_year = col2.selectbox('delivery year',[2021, 2022, 2020, 2019],key=21)
            delivery_month = col2.selectbox('delivery month',sorted([ 7,  4,  1,  3,  6,  5,  8,  9, 10, 11,  2, 12]),key=22)
            delivery_day = col2.selectbox('delivery day',[1],key=23)
            
            price = col3.text_input(f'Enter selling_price (Min:{df['selling_price'].min()}, Max:{df['selling_price'].max()})')
            quantity_tons = col3.text_input(f'Enter Quantity Tons (Min:{df["quantity tons"].min()}, Max:{df["quantity tons"].max()})')
            thickness = col3.text_input(f'Enter thickness (Min:{df["thickness"].min()}, Max:{df["thickness"].max()})')
            width = col3.text_input(f'Enter width(Min:1, (Min:{df["width"].min()}, Max:{df["width"].max()})')
            customer = col3.text_input(f'customer ID (Min:{df["customer"].min()}, Max:{df["customer"].max()})')
            #input_data=np.array([[54.151139,30156308.0,28.0,5.0,10.0,2.00,1500.0,5384.0,1.670799e+09,854.00,2021.0,4.0,1.0,2021.0,7.0,1.0]])
            input_data=np.array([[quantity_tons,customer,country,Numeric(item_type,"item_type"),application,thickness,width,Numeric(matrial,"matrial"),product_ref,price,item_year,item_month,item_day,delivery_year,delivery_month,delivery_day]])
            
            if st.form_submit_button(label='PREDICT STATUS'):

                    scaler_trans = pickle.load(open(r"C:\Users\user\Desktop\python\Copper_Industrial\Classification_scaler.pkl",'rb'))
                    result = pickle.load(open(r"C:\Users\user\Desktop\python\Copper_Industrial\RandomForestClassifier.pkl",'rb'))

                    scaled_new_data_point = scaler_trans.transform(input_data)
                    predicted_output = result.predict(scaled_new_data_point)
                    if predicted_output == 1:
                      st.write('Won')
                        
                    else:
                      if predicted_output == 0:
                          st.write('Lost')  
                      else:
                         st.write("Enter correct format")
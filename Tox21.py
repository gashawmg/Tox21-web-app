# import the necessary libraries
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from matplotlib import image, pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, confusion_matrix,recall_score,classification_report,roc_auc_score,roc_curve,auc,precision_recall_curve
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Avalon import pyAvalonTools
import mols2grid
import streamlit as st
import streamlit.components.v1 as components
import pickle
from PIL import Image
import base64
import io

 
#--------- Use trained lgbm classifier and XGBoost classifier 

with open('model_lgbmc.pkl','rb') as f:
         model1 = pickle.load(f)
with open('model_xgbc.pkl','rb') as f:
          model2 = pickle.load(f)
with open('scaler.pkl','rb') as f:
          scaler = pickle.load(f)

#----------------------------------------------------------------------
# hide 'Made with streamlit footer'
# reference:https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2
# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_page_config(page_title='Toxicity Prediction App',layout='wide')
st.sidebar.markdown('<h2 style="color:white;background-image: linear-gradient(to right, red , green);padding: 3%;border-radius:20px;text-align:center"> Use this Sidebar for Toxicity Prediction </h2>',unsafe_allow_html=True)


st.markdown('<h3 style="color:white;background-image: linear-gradient(to left, red , green);border-radius:20px;padding: 2%;text-align:center"> Web Application for Compound Toxicity Prediction </h3>',unsafe_allow_html=True)
#---------- Display my linkedin page on the sidebar and main page
st.markdown("""[Gashaw M.Goshu](https://www.linkedin.com/in/gashaw-m-goshu/), Ph.D in Organic Chemistry""")

#------------ Define toxicity and explain why it is important
st.markdown("""`Toxicity` is one of the most important parameters that needs to be addressed in drug discovery and development. It is the main cause of failure in drug discovery and development [reference](https://www.nature.com/articles/nrd2378#:~:text=Toxicity%20is%20a%20primary%20cause,are%20predictive%20of%20human%20toxicities.). Therefore, it is very important that we need to know or predict whether our compound is toxic or not before we invest too much time and money on it. This Web Appilication was developed using the [Tox21 data](http://bioinf.jku.at/research/DeepTox/tox21.html). The model is a binary classifier,which means it predicts whether a compound is toxic or not based on the compound's features (combination of 200 descriptors and 512 bits of Avalon fingerprints). The top two models were selected by training 5,235 compounds with known experimentaly determined toxicity data using 29 models. Best results were obtained by two models Light GBM Classifier(LGBMC) and XGBoost Classifier(XGBC) using accuracy as a metric in 10-fold cross-validation as shown below.""")

figure1 = Image.open('crossvalidation.jpg')
st.image(figure1, caption='Figure 1. 10-fold cross-validation using LGBMC and XGBC Classifiers on the 5,235 dataset')

 
st.markdown(""" The prediction on test dataset is based on consensus: if the two models (LGBMC & XGBC models) agreed that the compound is toxic in any of the 12 targets indicated the [NIH website](https://tripod.nih.gov/tox21/challenge/about.jsp), then the compound is classified as toxic otherwise the compound is classified as nontoxic. The accuracy of the consensus model on the test dataset was 77%. The prediction of the 472 test dataset is summerized as shown in the following **classification report and Receiver Operating Characteristic (ROC) curve**.""")

# Import test data that contains predicted values
test = pd.read_csv('tox21_test.csv')
test_descriptors_scaled = pd.read_csv('test_descriptors_scaled.csv').values
y_test_label = pd.read_csv('y_test_label.csv').values

st.text(classification_report(test.Actual.values,test.Predicted.values))

#----------- Plot ROC curves of test data using the two classifiers: 

LGBMC_probab_pred = model1.predict_proba(test_descriptors_scaled)

XGBC_probab = model2.predict_proba(test_descriptors_scaled)


fpr_xgbc, tpr_xgbc, thresholds_xgbc= roc_curve(y_test_label, XGBC_probab[:,1])

fpr_lgbm, tpr_lgbm, thresholds_lgbm= roc_curve(y_test_label, LGBMC_probab_pred[:,1])

# -------- Plot the figure of the test dataset on the webpage
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

auc_xgbc = auc(fpr_xgbc, tpr_xgbc)
plt.plot(fpr_lgbm, tpr_lgbm, label='Light GBM Classifier AUC (area = {:.2f})'.format(auc_lgbm))

plt.plot(fpr_xgbc, tpr_xgbc, label='XGBoost Classifier AUC (area = {:.2f})'.format(auc_xgbc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves using 687 input features')
plt.legend(loc='best')
plt.grid(True)
plt.show()
st.pyplot(plt)
st.markdown('Figure 2. True Positive vs False Positive rate at different classification thresholds of light GBM classifier and XG Boost classifier on the 472 test dataset.')
# plot test data on grid
st.markdown('<h3 style="border: 2px solid #4908d4;border-radius:10px;border-radius:20px;padding: 3%;text-align:center">  Each grid box contains the <i>structure of a compound, actual and predicted toxicity </i></h3>',unsafe_allow_html=True)
test_data = mols2grid.display(test,
                            subset=['img', 'Actual Property','Predicted Property'],
                            style={"Actual Property": lambda x: "color: red; font-weight: bold;" if x =='Toxic' else ""},
                             n_cols=5, n_rows=3,
                             
   
                            tooltip = ['Actual Property','Predicted Property'],fixedBondLength=25, clearBackground=False)._repr_html_()
components.html(test_data,height = 600,width=900, scrolling=False)

# ============ User input
data = st.sidebar.text_input('Enter SMILE Strings in single or double quotation separated by comma:',"['CCCCO']")
st.sidebar.markdown('''`or upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in 'SMILES' column:`''')
multi_data = st.sidebar.file_uploader("=====================================")

st.sidebar.markdown("""**If you upload your CSV file, click the button below to get the toxicity prediction** """)
prediction = st.sidebar.button('Predict Toxicity of Molecules')

#=========== custom style for submit button
m = st.markdown("""
<style>
div.stButton > button:first-child {
    border: 1px solid #2e048a;
    border-radius:10px;
}
</style>""", unsafe_allow_html=True)

#=========== Use these 196 molecular descriptors
descriptors_196 = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
       'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt',
       'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons',
       'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
       'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
       'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
       'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
       'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
       'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
       'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
       'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
       'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
       'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
       'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
       'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA',
       'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
       'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
       'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
       'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
       'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
       'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
       'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAromaticCarbocycles',
       'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
       'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
       'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
       'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO',
       'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N',
       'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
       'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
       'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
       'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
       'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
       'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
       'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
       'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
       'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
       'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
       'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
       'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
       'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',
       'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
       'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
       'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
       'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
       'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
       'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
       'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

#============= Remove salts from smiles
def remove_salt(smile):
    from rdkit.Chem.SaltRemover import SaltRemover
    try:
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(smile)
        res = remover.StripMol(mol)
        smi = Chem.MolToSmiles(res)
        return smi
    except ValueError:
        print("Oops! That was not a valid SMILES.")    

# ================= Get the names of the 200 descriptors from RDKit
def calc_rdkit2d_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    # Append 200 molecular descriptors to each molecule in a list
    Mol_descriptors =[]
    for mol in mols:
        # Calculate all 200 descriptors for each molecule
        mol=Chem.AddHs(mol)
        descriptors = np.array(calc.CalcDescriptors(mol))
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names  

#============ A function that can generate a csv file for output file to download
# Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/2
#           https://github.com/dataprofessor/ml-auto-app/blob/main/app.py
def filedownload(data,file):
    df = data.to_csv(index=False)
    f= base64.b64encode(df.encode()).decode()
    link = f'<a href ="data:file/csv; base64,{f}" download={file}> Download {file} file</a>'
    return link

if data!= "['CCCCO']":
    df = pd.DataFrame(eval(data), columns =['SMILES'])
    
    # Remove if there are any salts
    df['SMILES'].apply(remove_salt)

    #========= function call to calculate 200 molecular descriptors using SMILES
    Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df['SMILES'])

    #========= Put the 200 molecular descriptors in  table
    Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

    #========= Use only the 196 descriptors listed above
 
  # Dataset_with_200_descriptors.drop(columns=['MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge'],inplace=True)
    Dataset_with_200_descriptors = Dataset_with_200_descriptors[descriptors_196]
 
     # for each random split, calculate Avalon fingerprints for test set
    test_Avalon_fps = []
    for i in df['SMILES']:
        mol = Chem.MolFromSmiles(i) 
        test_Avalon_fps.append(pyAvalonTools.GetAvalonFP(mol))
        
    # For Avalon fingerprints
    test_X = np.array(test_Avalon_fps,dtype=float)
    df_test_Av_X = pd.DataFrame(test_X,columns =["Av{}".format(i) for i in range(1,513)])
    combined_test_descriptors = pd.concat([Dataset_with_200_descriptors,df_test_Av_X],axis=1)
     

    # Drop columns that are highly correlated (>0.97)
    drop_columns=['MaxAbsEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'BertzCT', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi3n', 'Chi4n', 'LabuteASA', 'EState_VSA1', 'VSA_EState1', 'HeavyAtomCount', 'NumHDonors', 'MolMR', 'fr_COO2', 'fr_Nhpyrrole', 'fr_benzene', 'fr_phenol', 'fr_phenol_noOrthoHbond']

    combined_test_descriptors.drop(columns=drop_columns,inplace=True)
    
    #======== The data was standardized using standard scaler
    test_scaled = scaler.transform(combined_test_descriptors)
   
    #---------------------------------------------------------------------

    #======== Prediction of toxicity using model1(LightGBM) and model2(XGBC)
    lgbm_preds = model1.predict(test_scaled)
    xgbc_preds = model2.predict(test_scaled)

    # Consensus predictions
    predicted=[]
    activity = []
    for i,j in zip(lgbm_preds,xgbc_preds):
        if (i==1 ) & (j==1):
            predicted.append(1.0)
            activity.append('Toxic')
        else:
            predicted.append(0.0) 
            activity.append('Nontoxic')
    
    df1 = pd.DataFrame(columns=['SMILES','Predicted','Predicted Property'])
    df1['SMILES'] =df['SMILES'].values
    df1['Predicted']= predicted
    df1['Predicted Property']=activity

    st.sidebar.write(df1)
 
    st.sidebar.markdown('''## See your output in the following table:''')
    #======= Display output with structure in table form
    # reference:https://github.com/dataprofessor/drugdiscovery
    raw_html = mols2grid.display(df1,
                            #subset=["Name", "img"],
                            subset=['img', 'Predicted Property'],
                            n_cols=5, n_rows=3,
                            tooltip = ['Predicted Property'],fixedBondLength=25, clearBackground=False)._repr_html_()
    components.html(raw_html, width=900, height=900, scrolling=False)

    #======= show CSV file attachment
    st.sidebar.markdown(filedownload(df1,"predicted_toxicity.csv"),unsafe_allow_html=True)

#===== Use uploaded SMILES to calculate their logS values
elif prediction:
     df2 = pd.read_csv(multi_data)
     df2['SMILES'].apply(remove_salt)
     Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df2['SMILES'])
     Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
     #========= Use only the 196 descriptors listed above
     Dataset_with_200_descriptors = Dataset_with_200_descriptors[descriptors_196]
    
     X_test = Dataset_with_200_descriptors
     #Dataset_with_200_descriptors.drop(columns=['MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge'],inplace=True)
     test_Avalon_fps = []
     for i in df2['SMILES']:
        mol = Chem.MolFromSmiles(i) 
        test_Avalon_fps.append(pyAvalonTools.GetAvalonFP(mol))
    # For Avalon fingerprints
     test_X = np.array(test_Avalon_fps,dtype=float)
     df_test_Av_X = pd.DataFrame(test_X,columns =["Av{}".format(i) for i in range(1,513)])
     combined_test_descriptors = pd.concat([Dataset_with_200_descriptors,df_test_Av_X],axis=1)


    # Drop columns that are highly correlated (>0.97)
     drop_columns=['MaxAbsEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'BertzCT', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi3n', 'Chi4n', 'LabuteASA', 'EState_VSA1', 'VSA_EState1', 'HeavyAtomCount', 'NumHDonors', 'MolMR', 'fr_COO2', 'fr_Nhpyrrole', 'fr_benzene', 'fr_phenol', 'fr_phenol_noOrthoHbond']

     combined_test_descriptors.drop(columns=drop_columns,inplace=True)
    
    #======== The data was standardized using standard scaler
     test_scaled = scaler.transform(combined_test_descriptors)

    #---------------------------------------------------------------------

    #======== Prediction of toxicity using model1(LightGBM) and model2(XGBC)
     lgbm_preds = model1.predict(test_scaled)
     xgbc_preds = model2.predict(test_scaled)

    # Consensus predictions
     predicted=[]
     activity = []
     for i,j in zip(lgbm_preds,xgbc_preds):
        if (i==1 ) & (j==1):
            predicted.append(1.0)
            activity.append('Toxic')
        else:
            predicted.append(0.0) 
            activity.append('Nontoxic')
    
     df3 = pd.DataFrame(columns=['SMILES','Predicted','Predicted Property'])
     df3['SMILES'] = df2['SMILES'].values
     df3['Predicted'] = predicted
     df3['Predicted Property'] = activity

     st.sidebar.markdown('''## See your output in the following table:''')
    #======= Display output in table form
     st.sidebar.write(df3)

    #======= show CSV file attachment
     st.sidebar.markdown(filedownload(df3,"predicted_toxicity.csv"),unsafe_allow_html=True)
   # st.markdown('''## See the output shown below:''')

    #======= Display output with structure in table form
    # reference:https://github.com/dataprofessor/drugdiscovery
     raw_html = mols2grid.display(df3,
                            #subset=["Name", "img"],
                            subset=['img', 'Predicted Property'],
                            n_cols=5, n_rows=3,
                            tooltip = ['Predicted Property'],fixedBondLength=25, clearBackground=False)._repr_html_()
     components.html(raw_html,width=900, height=900, scrolling=False)

else:
    st.markdown('<div style="border: 2px solid #4908d4;border-radius:20px;padding: 3%;text-align:center"><h5> If you want to test this model,  please use the sidebar. If you have few molecules, you can directly put the SMILES in a single or double quotations separated by comma in the sidebar. If you have many molecules, you can put their SMILES strings in a "SMILES" column, upload them and click the button which says "Predict Toxicity of Molecules" shown in the sidebar.</h5> <h5 style="color:white;background-color:#0a0605;border-radius:10px;padding: 3%;opacity: 0.7;">Please also note that predcition is more reliable if the compounds to be predicted are similar with training dataset</h5></div>',unsafe_allow_html=True)
   

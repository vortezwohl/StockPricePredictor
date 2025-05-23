import pandas as pd

dow_jones = pd.read_csv('./data/train/indices/Dow_Jones.csv').to_dict(orient='records')
nasdaq = pd.read_csv('./data/train/indices/NASDAQ.csv').to_dict(orient='records')
sp500 = pd.read_csv('./data/train/indices/SP500.csv').to_dict(orient='records')

stockAAPL = pd.read_csv('./data/train/stocks/AAPL.csv').to_dict(orient='records')
stockABBV = pd.read_csv('./data/train/stocks/ABBV.csv').to_dict(orient='records')
stockAXP = pd.read_csv('./data/train/stocks/AXP.csv').to_dict(orient='records')
stockBA = pd.read_csv('./data/train/stocks/BA.csv').to_dict(orient='records')
stockBOOT = pd.read_csv('./data/train/stocks/BOOT.csv').to_dict(orient='records')
stockCALM = pd.read_csv('./data/train/stocks/CALM.csv').to_dict(orient='records')
stockCAT = pd.read_csv('./data/train/stocks/CAT.csv').to_dict(orient='records')
stockCL = pd.read_csv('./data/train/stocks/CL.csv').to_dict(orient='records')
stockCSCO = pd.read_csv('./data/train/stocks/CSCO.csv').to_dict(orient='records')
stockCVX = pd.read_csv('./data/train/stocks/CVX.csv').to_dict(orient='records')
stockDD = pd.read_csv('./data/train/stocks/DD.csv').to_dict(orient='records')
stockDENN = pd.read_csv('./data/train/stocks/DENN.csv').to_dict(orient='records')
stockDIS = pd.read_csv('./data/train/stocks/DIS.csv').to_dict(orient='records')
stockF = pd.read_csv('./data/train/stocks/F.csv').to_dict(orient='records')
stockGE = pd.read_csv('./data/train/stocks/GE.csv').to_dict(orient='records')
stockGM = pd.read_csv('./data/train/stocks/GM.csv').to_dict(orient='records')
stockGS = pd.read_csv('./data/train/stocks/GS.csv').to_dict(orient='records')
stockHON = pd.read_csv('./data/train/stocks/HON.csv').to_dict(orient='records')
stockIBM = pd.read_csv('./data/train/stocks/IBM.csv').to_dict(orient='records')
stockINTC = pd.read_csv('./data/train/stocks/INTC.csv').to_dict(orient='records')
stockIP = pd.read_csv('./data/train/stocks/IP.csv').to_dict(orient='records')
stockJNJ = pd.read_csv('./data/train/stocks/JNJ.csv').to_dict(orient='records')
stockJPM = pd.read_csv('./data/train/stocks/JPM.csv').to_dict(orient='records')
stockKO = pd.read_csv('./data/train/stocks/KO.csv').to_dict(orient='records')
stockLMT = pd.read_csv('./data/train/stocks/LMT.csv').to_dict(orient='records')
stockMA = pd.read_csv('./data/train/stocks/MA.csv').to_dict(orient='records')
stockMCD = pd.read_csv('./data/train/stocks/MCD.csv').to_dict(orient='records')
stockMG = pd.read_csv('./data/train/stocks/MG.csv').to_dict(orient='records')
stockMMM = pd.read_csv('./data/train/stocks/MMM.csv').to_dict(orient='records')
stockMS = pd.read_csv('./data/train/stocks/MS.csv').to_dict(orient='records')
stockMSFT = pd.read_csv('./data/train/stocks/MSFT.csv').to_dict(orient='records')
stockNKE = pd.read_csv('./data/train/stocks/NKE.csv').to_dict(orient='records')
stockPEP = pd.read_csv('./data/train/stocks/PEP.csv').to_dict(orient='records')
stockPFE = pd.read_csv('./data/train/stocks/PFE.csv').to_dict(orient='records')
stockPG = pd.read_csv('./data/train/stocks/PG.csv').to_dict(orient='records')
stockRTX = pd.read_csv('./data/train/stocks/RTX.csv').to_dict(orient='records')
stockSO = pd.read_csv('./data/train/stocks/SO.csv').to_dict(orient='records')
stockT = pd.read_csv('./data/train/stocks/T.csv').to_dict(orient='records')
stockTDW = pd.read_csv('./data/train/stocks/TDW.csv').to_dict(orient='records')
stockV = pd.read_csv('./data/train/stocks/V.csv').to_dict(orient='records')
stockVZ = pd.read_csv('./data/train/stocks/VZ.csv').to_dict(orient='records')
stockWFC = pd.read_csv('./data/train/stocks/WFC.csv').to_dict(orient='records')
stockWMT = pd.read_csv('./data/train/stocks/WMT.csv').to_dict(orient='records')
stockXELB = pd.read_csv('./data/train/stocks/XELB.csv').to_dict(orient='records')
stockXOM = pd.read_csv('./data/train/stocks/XOM.csv').to_dict(orient='records')

dataset = [
    dow_jones,
    nasdaq,
    sp500,
    stockAAPL,
    stockABBV,
    stockAXP,
    stockBA,
    stockBOOT,
    stockCALM,
    stockCAT,
    stockCL,
    stockCSCO,
    stockCVX,
    stockDD,
    stockDENN,
    stockDIS,
    stockF,
    stockGE,
    stockGM,
    stockGS,
    stockHON,
    stockIBM,
    stockINTC,
    stockIP,
    stockJNJ,
    stockJPM,
    stockKO,
    stockLMT,
    stockMA,
    stockMCD,
    stockMG,
    stockMMM,
    stockMS,
    stockMSFT,
    stockNKE,
    stockPEP,
    stockPFE,
    stockPG,
    stockRTX,
    stockSO,
    stockT,
    stockTDW,
    stockV,
    stockVZ,
    stockWFC,
    stockWMT,
    stockXELB,
    stockXOM
]

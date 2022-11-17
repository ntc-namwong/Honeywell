'''
YOKOGAWA APC MODEL CONVERTER
============================
This main purpose is to convert Yokogawa APC to Generic Model for CPM APC configuration.
This software was built on 1 July 2022 by Nathachok Namwong, Honeywell Thailand.
'''

# %% Import library
import os
import control
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from tkinter import ttk, filedialog
from collections import defaultdict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# %% Controller

class Controller:
    def __init__(self, ControllerType, ModelFile, MappingFile, excEF=True):
        '''
        Controller class contains MV, DV, CV and POV list as well as their relationship between each variables based on Model file.
        
        Parameters
        ----------
        ControllerType : string
            Controller type can be either SMOC or PACE. Other types are currently not supported yet.
        ModelFile : string
            ModelFile is file location of model.
            - (SMOC) := htm file
            - (PACE) := xml file
        MappingFile   : string
            InputFile is file location of tag mapping.
            - (SMOC) := MV Detail, DV Detail, CV Detail, POV Detail, Sub Controller
            - (PACE) := Input, Output, Datalink
        excEF : boolean
            excEF is used for excluding economic function variables, especially the variable named contained MIN or MAX.
            True = excluded, False = included
        '''
        # Initializing
        self.ControllerType = ControllerType
        self.ModelPath = list()
        self.FinalModel = list()
        self.VariablePath = list()
        
        # SMOC controller
        if self.ControllerType == 'SMOC':
            # Loading model and variables
            if ModelFile.endswith('.htm'):
                self.MVs, self.DVs, self.CVs, self.POVs, self.Models, self.Mappings = extractHTML(ModelFile, MappingFile)
        # PACE controller
        elif self.ControllerType == 'PACE':
            # Loading model
            if ModelFile.endswith('.xml'):
                self.MVs, self.DVs, self.CVs, self.POVs, self.Models, self.Mappings = extractXML(ModelFile, MappingFile)

        # Excluding economic function variables 
        if excEF:
            self.CVs  = {i:cv for i, cv in self.CVs.items() if 'min' not in cv.lower()}
            self.CVs  = {i:cv for i, cv in self.CVs.items() if 'max' not in cv.lower()}
            self.CVs  = {i:cv for i, cv in self.CVs.items() if '-ll' not in cv.lower()}
            self.CVs  = {i:cv for i, cv in self.CVs.items() if '-hh' not in cv.lower()}
            self.POVs = {i:pov for i, pov in self.POVs.items() if 'min' not in pov.lower()}
            self.POVs = {i:pov for i, pov in self.POVs.items() if 'max' not in pov.lower()}
            self.POVs = {i:pov for i, pov in self.POVs.items() if '-ll' not in pov.lower()}
            self.POVs = {i:pov for i, pov in self.POVs.items() if '-hh' not in pov.lower()}
        
        # Reseting index
        self.resetIndex('ALL')
        
        # Duplicating original variables
        self.MVo  = self.MVs.copy()
        self.DVo  = self.DVs.copy()
        self.CVo  = self.CVs.copy()
        self.POVo = self.POVs.copy()
        
    def resetIndex(self, VarType):
        if VarType.upper() == 'ALL':
            self.MVs = {i: v for i, v in enumerate(self.MVs.values())}
            self.DVs = {i: v for i, v in enumerate(self.DVs.values())}
            self.CVs = {i: v for i, v in enumerate(self.CVs.values())}
            self.POVs = {i: v for i, v in enumerate(self.POVs.values())}
        if VarType.upper() == 'MV':
            self.MVs = {i: v for i, v in enumerate(self.MVs.values())}
        if VarType.upper() == 'DV':
            self.DVs = {i: v for i, v in enumerate(self.DVs.values())}
        if VarType.upper() == 'CV':
            self.CVs = {i: v for i, v in enumerate(self.CVs.values())}
        if VarType.upper() == 'POV':
            self.POVs = {i: v for i, v in enumerate(self.POVs.values())}
    
    def moveUp(self, VarType, Current):
        if VarType.upper() == 'MV':
            self.MVs[Current-1], self.MVs[Current] = self.MVs[Current], self.MVs[Current-1]
        if VarType.upper() == 'DV':
            self.DVs[Current-1], self.DVs[Current] = self.DVs[Current], self.DVs[Current-1]
        if VarType.upper() == 'CV':
            self.CVs[Current-1], self.CVs[Current] = self.CVs[Current], self.CVs[Current-1]
        if VarType.upper() == 'POV':
            self.POVs[Current-1], self.POVs[Current] = self.POVs[Current], self.POVs[Current-1]
    
    def moveDn(self, VarType, Current):
        if VarType.upper() == 'MV':
            self.MVs[Current+1], self.MVs[Current] = self.MVs[Current], self.MVs[Current+1]
        if VarType.upper() == 'DV':
            self.DVs[Current+1], self.DVs[Current] = self.DVs[Current], self.DVs[Current+1]
        if VarType.upper() == 'CV':
            self.CVs[Current+1], self.CVs[Current] = self.CVs[Current], self.CVs[Current+1]
        if VarType.upper() == 'POV':
            self.POVs[Current+1], self.POVs[Current] = self.POVs[Current], self.POVs[Current+1]
    
    def moveMV2DV(self, Current, New):
        self.DVs[New] = self.MVs[Current]
        self.MVs.pop(Current, None)
    
    def moveDV2MV(self, Current, New):
        self.MVs[New] = self.DVs[Current]
        self.DVs.pop(Current, None)
        
    def moveDV2POV(self, Current, New):
        self.POVs[New] = self.DVs[Current]
        self.DVs.pop(Current, None)
    
    def moveCV2POV(self, Current, New):
        self.POVs[New] = self.CVs[Current]
        self.CVs.pop(Current, None)
    
    def movePOV2DV(self, Current, New):
        self.DVs[New] = self.POVs[Current]
        self.POVs.pop(Current, None)
        
    def movePOV2CV(self, Current, New):
        self.CVs[New] = self.POVs[Current]
        self.POVs.pop(Current, None)
        
    def findPath(self, u, d, Visited, Path, Model, Graph):
        '''
        To find all possible path from MV/DV to CV

        Parameters
        ----------
        u : string
            Variable name of the starting point
        d : string
            Variable name of the destination
        visited : dictionary of boolean
            Variable that has been used
        Path : list
            Variable path
        Model : list
            Model path
        Graph : dictionary of tuple
            Variable graph
        '''
        Visited[u] = True
        Path.append(u)
        if u == d:
            self.VariablePath.append(tuple(Path))
            self.ModelPath.append(tuple(Model))
        else:
            for i, m in Graph[u]:
                if Visited[i] == False:
                    Model.append(m)
                    self.findPath(i, d, Visited, Path, Model, Graph)
        Visited[u] = False
        if len(Path) > 0:  Path.pop()
        if len(Model) > 0: Model.pop()
        
    def buildModel(self):
        '''
        To build final model from SMOC/PACE submodel
        
        Returns
        -------
        FinalModel : list of (string, string, int, int, list, list)
            Final model includes 1) input variable name, 2) output variable name, 3) gain, 4) delay, 5) numerator, and 6) denominator
        '''
        # Initializing
        Bond = list()
        
        # CV-MV relationship
        for i in self.MVs.values():
            for o in self.CVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # CV-DV relationship
        for i in self.DVs.values():
            for o in self.CVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # POV-MV Relationship
        for i in self.MVs.values():
            for o in self.POVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # POV-DV Relationship
        for i in self.DVs.values():
            for o in self.POVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # POV-CV relationship
        for i in self.CVs.values():
            for o in self.POVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # POV-POV relationship
        for i in self.POVs.values():
            for o in self.POVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # CV-CV relationship
        for i in self.CVs.values():
            for o in self.CVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # CV-POV relationship
        for i in self.POVs.values():
            for o in self.CVs.values():
                idx = self.Models.index[(self.Models['InputVariable'] == i) &
                                        (self.Models['OutputVariable'] == o)].to_list()
                if len(idx) > 0:
                    Bond.append((i, o, idx[0]))
        
        # All path relationship
        Graph = defaultdict(list)
        for b in Bond:
            Graph[b[0]].append((b[1], b[2]))
            Graph[b[1]].append((b[0], b[2]))
        
        # Individual path
        for mv in self.MVs.values():
            for cv in self.CVs.values():
                Visited = {k:False for k in Graph.keys()}
                self.findPath(mv, cv, Visited, [], [], Graph)
        for dv in self.DVs.values():
            for cv in self.CVs.values():
                Visited = {k:False for k in Graph.keys()}
                self.findPath(dv, cv, Visited, [], [], Graph)
        
        # Final model
        for var, mdl in zip(self.VariablePath, self.ModelPath):
            for i, _ in enumerate(var):
                if i == 0:
                    pass
                elif i == 1:
                    m = Model(var[i-1], var[i], mdl[i-1], self.Models)
                else:
                    m = multMDL(m, Model(var[i-1], var[i], mdl[i-1], self.Models))
            self.FinalModel.append((m.Input, m.Output, m.Gain, m.Delay, m.Numerator, m.Denominator))

class Model:
    def __init__(self, Input = '', Output = '', idx = 0, Model = None):
        '''
        Model class contains relationship between input and output variables.

        Parameters
        ----------
        Input : string, optional
            Input variable name. The default is ''.
        Output : string, optional
            Output variable name. The default is ''.
        idx : int, optional
            Model index in the Models DataFrame. The default is 0.
        Model : DataFrame, optional
            Model DataFrame. The default is None.
        '''
        # Initializing
        self.Input = Input
        self.Output = Output
        
        if Model is not None:
            self.Gain = float(Model.iloc[idx]['Gain'])
            self.Delay = float(Model.iloc[idx]['Delay'])
            if Model.iloc[idx]['Order'] == 'Gain':
                self.Numerator = [1]
                self.Denominator = [1]
            elif Model.iloc[idx]['Order'] == 'FirstOrder' or Model.iloc[idx]['Order'] == 'First Order':
                self.Numerator = [1]
                self.Denominator = [1, float(Model.iloc[idx]['Tau 1'])]
            elif Model.iloc[idx]['Order'] == 'Ramp':
                self.Numerator = [1]
                self.Denominator = [0, 1, float(Model.iloc[idx]['Tau 1'])]
            elif Model.iloc[idx]['Order'] == 'FirstOrderRamp':
                self.Numerator = [1, float(Model.iloc[idx]['Beta'])]
                self.Denominator = [0, float(Model.iloc[idx]['Tau 1']), float(Model.iloc[idx]['Tau 2'])]
            elif Model.iloc[idx]['Order'] == 'SecondOrderBeta' or Model.iloc[idx]['Order'] == 'Second Order':
                self.Numerator = [1, float(Model.iloc[idx]['Beta'])]
                self.Denominator = [1, float(Model.iloc[idx]['Tau 1']), float(Model.iloc[idx]['Tau 2'])]
            elif Model.iloc[idx]['Order'] == 'ZeroGainSecondOrder':
                self.Numerator = [0, float(Model.iloc[idx]['Beta'])]
                self.Denominator = [1, float(Model.iloc[idx]['Tau 1']), float(Model.iloc[idx]['Tau 2'])]
        else:
            self.Gain = 0
            self.Dely = 0
            self.Numerator = []
            self.Denominator = []

def extractXML(ModelFile, MappingFile):
    '''
    Extract XML function is used to extract PACE model from the XML file.
    XML file must select Application as highest hierarchy.

    Parameters
    ----------
    ModelFile : string
        location of the xml model file

    Returns
    -------
    Models : DataFrame
        | InputVariable | OutputVariable | Order | Gain | Tau 1 | Tau 2 | Beta | Delay |
    '''
    # Initializing
    Models = list()
    xmlns = '{http://www.shell.com/APC/QuantumLeap}'
    
    # Read Excel file
    dfInput  = pd.read_excel(MappingFile, sheet_name = 'Input')
    dfOutput = pd.read_excel(MappingFile, sheet_name = 'Output')
    dfLink = pd.read_excel(MappingFile, sheet_name = 'Datalink')
    
    # Variables
    MVs  = dfInput.loc[(dfInput['Ind'] == 'I') & (dfInput['MV'] == True) ]['Name'].to_dict()
    DVs  = dfInput.loc[(dfInput['Ind'] == 'I') & (dfInput['MV'] == False)]['Name'].to_dict()
    CVs  = dfOutput.loc[dfOutput['CV'] == True ]['Name'].to_dict()
    POVs = dfOutput.loc[dfOutput['CV'] == False]['Name'].to_dict()
    
    # Tag mapping
    Mapping = dict()
    Mapping['Name'] = ['Controller']
    Mapping['Tag']  = ['']
    
    # Mode
    try:
        Mapping['Mode'] = [dfLink.loc[dfLink['Reference'] == 'ModeActual', 'Reference.1'].item()]
    except ValueError:
        Mapping['Mode'] = ['']
    
    Mapping['High'] = ['']
    Mapping['Low']  = ['']
    Mapping['SST']  = ['']
    Mapping['PredErr']  = ['']
    Mapping['OptMode']  = ['']
    Mapping['OptCoeff'] = ['']
    Mapping['MaxMove']  = ['']
    Mapping['WindUp']   = ['']
    
    # MV
    for mv in MVs.values():
        Mapping['Name'].append(mv)
        Mapping['Tag'].append(mv)
        
        # Mode
        try:
            Mapping['Mode'].append(dfLink.loc[dfLink['Reference'] == f'Variables/{mv}/MV/ModeActual', 'Reference.1'].item())
        except ValueError:
            Mapping['Mode'].append('')
        
        # High limit
        try:
            Mapping['High'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{mv}/MV/HiLimit', 'Reference'].item())
        except ValueError:
            Mapping['High'].append('')
        
        # Low limit
        try:
            Mapping['Low'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{mv}/MV/LoLimit', 'Reference'].item())
        except ValueError:
            Mapping['Low'].append('')
        
        # SST
        try:
            Mapping['SST'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{mv}/MV/SteadyStateValue', 'Reference'].item())
        except ValueError:
            Mapping['SST'].append('')
        
        Mapping['PredErr'].append('')
        Mapping['OptMode'].append('')
        Mapping['OptCoeff'].append('')
        
        # Max move
        try:
            Mapping['MaxMove'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{mv}/MV/MaxMoveSizeUp', 'Reference'].item())
        except ValueError:
            Mapping['MaxMove'].append('')
        
        Mapping['WindUp'].append('')
    
    # DV
    for dv in DVs.values():
        Mapping['Name'].append(dv)
        Mapping['Tag'].append(dv)
        
        # Mode
        try:
            Mapping['Mode'].append(dfLink.loc[dfLink['Reference'] == f'Variables/{dv}/DV/ModeActual', 'Reference.1'].item())
        except ValueError:
            Mapping['Mode'].append('')
        
        Mapping['High'].append('')
        Mapping['Low'].append('')
        Mapping['SST'].append('')
        Mapping['PredErr'].append('')
        Mapping['OptMode'].append('')
        Mapping['OptCoeff'].append('')
        Mapping['MaxMove'].append('')
        Mapping['WindUp'].append('')
        
    # CV
    for cv in CVs.values():
        Mapping['Name'].append(cv)
        Mapping['Tag'].append(cv)
        
        # Mode
        try:
            Mapping['Mode'].append(dfLink.loc[dfLink['Reference'] == f'Variables/{cv}/CV/ModeActual', 'Reference.1'].item())
        except ValueError:
            Mapping['Mode'].append('')
        
        # High limit
        try:
            Mapping['High'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{cv}/CV/HiLimit', 'Reference'].item())
        except ValueError:
            Mapping['High'].append('')
        
        # Low limit
        try:
            Mapping['Low'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{cv}/CV/LoLimit', 'Reference'].item())
        except ValueError:
            Mapping['Low'].append('')
        
        # SST
        try:
            Mapping['SST'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{cv}/CV/SteadyStateValue', 'Reference'].item())
        except ValueError:
            Mapping['SST'].append('')
        
        # PredErr
        try:
            Mapping['PredErr'].append(dfLink.loc[dfLink['Reference'] == f'Variables/{cv}/POV/PredictedValue', 'Reference.1'].item())
        except ValueError:
            Mapping['PredErr'].append('')
        
        Mapping['OptMode'].append('')
        Mapping['OptCoeff'].append('')
        Mapping['MaxMove'].append('')
        Mapping['WindUp'].append('')
    
    # POV
    for pov in POVs.values():
        Mapping['Name'].append(pov)
        Mapping['Tag'].append(pov)
        
        # Mode
        try:
            Mapping['Mode'].append(dfLink.loc[dfLink['Reference'] == f'Variables/{pov}/POV/ModeActual', 'Reference.1'].item())
        except ValueError:
            Mapping['Mode'].append('')
        
        # High limit
        try:
            Mapping['High'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{pov}/POV/HiLimit', 'Reference'].item())
        except ValueError:
            Mapping['High'].append('')
        
        # Low limit
        try:
            Mapping['Low'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{pov}/POV/LoLimit', 'Reference'].item())
        except ValueError:
            Mapping['Low'].append('')
        
        # SST
        try:
            Mapping['SST'].append(dfLink.loc[dfLink['Reference.1'] == f'Variables/{pov}/POV/SteadyStateValue', 'Reference'].item())
        except ValueError:
            Mapping['SST'].append('')
        
        # PredErr
        try:
            Mapping['PredErr'].append(dfLink.loc[dfLink['Reference'] == f'Variables/{pov}/POV/PredictedValue', 'Reference.1'].item())
        except ValueError:
            Mapping['PredErr'].append('')
        
        Mapping['OptMode'].append('')
        Mapping['OptCoeff'].append('')
        Mapping['MaxMove'].append('')
        Mapping['WindUp'].append('')
    
    Mappings = pd.DataFrame.from_dict(Mapping)
    
    # Variable list
    tree = ET.parse(ModelFile)
    
    # SISO model
    blocks = tree.find(xmlns + 'Models/' + xmlns + 'Model/' + xmlns + 'Blocks')
    for b in blocks:
        b = b.find(xmlns + 'SisoRelationship')
        Models.append([b.find(xmlns + 'InputVariable').text,
                      b.find(xmlns + 'OutputVariable').text,
                      b.find(xmlns + 'ActiveElement/' + xmlns + 'TransferFunction').text,
                      b.find(xmlns + 'ActiveElement/' + xmlns + 'Gain').text,
                      str(float(b.find(xmlns + 'ActiveElement/' + xmlns + 'Tau1').text)/60),
                      str(float(b.find(xmlns + 'ActiveElement/' + xmlns + 'Tau2').text)/60),
                      str(float(b.find(xmlns + 'ActiveElement/' + xmlns + 'Beta').text)/60),
                      str(float(b.find(xmlns + 'ActiveElement/' + xmlns + 'Delay').text)/60)])
    Models = pd.DataFrame(Models, columns=['InputVariable', 'OutputVariable', 'Order',
                                           'Gain', 'Tau 1', 'Tau 2', 'Beta', 'Delay'])
    Models = Models.astype({'InputVariable': str, 'OutputVariable': str, 'Order': str, 'Gain': float,
                            'Tau 1': float, 'Tau 2': float, 'Beta': float, 'Delay': float})
    
    return MVs, DVs, CVs, POVs, Models, Mappings

def extractHTML(ModelFile, MappingFile):
    '''
    Extract HTML function is used to extract SMOC model from the HTML file.
    HTML file must select Application as highest hierarchy.

    Parameters
    ----------
    ModelFile : string
        location of the html model file
    MappingFile : string
        location of Excel MV, DV, CV, POV and Sub Model detail

    Returns
    -------
    Models : DataFrame
        | InputVariable | OutputVariable | Order | Gain | Tau 1 | Tau 2 | Beta | Delay |
    '''
    # Read HTM file
    df_model = pd.read_html(ModelFile)[1]
    df_model = df_model.rename(columns=df_model.iloc[0]).drop(df_model.index[0])
    df_model = df_model.dropna(subset=['Model'])
    
    # Variables
    MVs = df_model[df_model['Model'].str.contains('MV')]['Block Name'].to_dict()
    DVs = df_model[df_model['Model'].str.contains('DV')]['Block Name'].to_dict()
    CVs = dict()
    POVs = df_model[df_model['Model'].str.contains('Output')]['Block Name'].to_dict()
    
    # Read Excel file
    df_mv = pd.read_excel(MappingFile, sheet_name = 'MV Detail')
    df_dv = pd.read_excel(MappingFile, sheet_name = 'DV Detail')
    df_cv = pd.read_excel(MappingFile, sheet_name = 'CV Detail')
    df_pov = pd.read_excel(MappingFile, sheet_name = 'POV Detail')
    df_sub = pd.read_excel(MappingFile, sheet_name = 'Sub Controller')
    
    # Tag mapping
    Mapping = dict()
    Mapping['Name'] = ['Controller']
    Mapping['Tag']  = ['']
    Mapping['Mode'] = [df_sub.loc[df_sub['Alias'] == 'Actual Status [t/v]'].iloc[:, 1].item()]
    Mapping['High'] = ['']
    Mapping['Low']  = ['']
    Mapping['SST']  = ['']
    Mapping['PredErr']  = ['']
    Mapping['OptMode']  = ['']
    Mapping['OptCoeff'] = ['']
    Mapping['MaxMove']  = ['']
    Mapping['WindUp']   = ['']
    
    for mv in MVs.values():
        Mapping['Name'].append(mv)
        Mapping['Tag'].append(df_mv.loc[df_mv['Alias'] == 'Setpoint Readback Tag [t]', mv].item())
        Mapping['Mode'].append(df_mv.loc[df_mv['Alias'] == 'Remote / Local Flag Tag [t]', mv].item())
        Mapping['High'].append(df_mv.loc[df_mv['Alias'] == 'Setpoint High Limit [t/v]', mv].item())
        Mapping['Low'].append(df_mv.loc[df_mv['Alias'] == 'Setpoint Low Limit [t/v]', mv].item())
        Mapping['SST'].append(df_mv.loc[df_mv['Alias'] == 'Steady State Reachable Value [t/v]', mv].item())
        Mapping['PredErr'].append('')
        Mapping['OptMode'].append('')
        Mapping['OptCoeff'].append('')
        Mapping['MaxMove'].append(df_mv.loc[df_mv['Alias'] == 'Maximum Move Size [t/v]', mv].item())
        Mapping['WindUp'].append('')
    
    for dv in DVs.values():
        Mapping['Name'].append(dv)
        Mapping['Tag'].append(df_dv.loc[df_dv['Alias'] == 'Measurement Tag [t]', dv].item())
        Mapping['Mode'].append(df_dv.loc[df_dv['Alias'] == 'Disconnect Flag [t/v]', dv].item())
        Mapping['High'].append('')
        Mapping['Low'].append('')
        Mapping['SST'].append('')
        Mapping['PredErr'].append('')
        Mapping['OptMode'].append('')
        Mapping['OptCoeff'].append('')
        Mapping['MaxMove'].append('')
        Mapping['WindUp'].append('')
    
    POVtemp = POVs.copy()
    for i, pov in POVtemp.items():
        if pov in df_cv.columns:
            CVs[i] = POVs.pop(i)
            Mapping['Name'].append(pov)
            Mapping['Tag'].append(df_pov.loc[df_pov['Alias'] == 'Measurement Tag [t]', pov].item())
            Mapping['Mode'].append(df_cv.loc[df_cv['Alias'] == 'Actual CV Status [t/v]', pov].item())
            Mapping['High'].append(df_cv.loc[df_cv['Alias'] == 'SetRange High [t/v]', pov].item())
            Mapping['Low'].append(df_cv.loc[df_cv['Alias'] == 'SetRange Low [t/v]', pov].item())
            Mapping['SST'].append(df_cv.loc[df_cv['Alias'] == 'Steady State Reachable Value [t/v]', pov].item())
            Mapping['PredErr'].append(df_cv.loc[df_cv['Alias'] == 'Calculated Value [t/v]', pov].item())
            Mapping['OptMode'].append('')
            Mapping['OptCoeff'].append('')
            Mapping['MaxMove'].append('')
            Mapping['WindUp'].append('')
        else:
            Mapping['Name'].append(pov)
            Mapping['Tag'].append(df_pov.loc[df_pov['Alias'] == 'Measurement Tag [t]', pov].item())
            Mapping['Mode'].append('')
            Mapping['High'].append('')
            Mapping['Low'].append('')
            Mapping['SST'].append('')
            Mapping['PredErr'].append('')
            Mapping['OptMode'].append('')
            Mapping['OptCoeff'].append('')
            Mapping['MaxMove'].append('')
            Mapping['WindUp'].append('')
    
    Mappings = pd.DataFrame.from_dict(Mapping)
    
    # Reset index
    try:
        POVs = {i: v for i, v in enumerate(POVs.values())}
        CVs  = {i: v for i, v in enumerate(CVs.values())}
    except:
        pass
    
    # Models
    Models = df_model[~(df_model['Model'].str.contains('Input') | df_model['Model'].str.contains('Output'))].reset_index()
    Models = Models[['Input', 'Output', 'Model', 'Gain', 'Tau 1', 'Tau 2', 'Beta', 'Dead Time']].fillna(0.0)
    Models = Models.astype({'Input': str, 'Output': str, 'Model': str, 'Gain': float,
                            'Tau 1': float, 'Tau 2': float, 'Beta': float, 'Dead Time': float})
    Models.columns = ['InputVariable', 'OutputVariable', 'Order', 'Gain', 'Tau 1', 'Tau 2', 'Beta', 'Delay']
    
    return MVs, DVs, CVs, POVs, Models, Mappings

def multMDL(mdl1, mdl2):
    '''
    Model multiplication based on Laplace transform form.

    Parameters
    ----------
    mdl1 : Model Class
        Model no 1
    mdl2 : Model Class
        Model no 2

    Raises
    ------
    NoRelationship
        Model multiplication can be used only if output of one model is same as input of another model

    Returns
    -------
    result : Model Class
        Result model
    '''
    # Validation
    if mdl1.Output == mdl2.Input:
        result = Model()
        result.Input = mdl1.Input
        result.Output = mdl2.Output
    elif mdl1.Input == mdl2.Output:
        result = Model()
        result.Input = mdl2.Input
        result.Output = mdl1.Output
    
    # Gain
    result.Gain  = mdl1.Gain * mdl2.Gain
    
    # Delay
    result.Delay = mdl1.Delay + mdl2.Delay
    
    # Numerator and Denominator
    for i in range(len(mdl1.Numerator) + len(mdl2.Numerator) - 1):
        result.Numerator.append(0)
    for i in range(len(mdl1.Denominator) + len(mdl2.Denominator) - 1):
        result.Denominator.append(0)
    for i1 in range(len(mdl1.Numerator)):
        for i2 in range(len(mdl2.Numerator)):
            result.Numerator[i1+i2] = result.Numerator[i1+i2] + mdl1.Numerator[i1]*mdl2.Numerator[i2]
    for i1 in range(len(mdl1.Denominator)):
        for i2 in range(len(mdl2.Denominator)):
            result.Denominator[i1+i2] = result.Denominator[i1+i2] + mdl1.Denominator[i1]*mdl2.Denominator[i2]
    
    return result
        
# %% Graphic User Interface

class App(ttk.Frame):
    def __init__(self, parent):
        '''
        Yokogawa APC model converter graphic user interface contains 3 main pages, including
        page 1 - Input, page 2 - Preprocessing, and page 3 - Output

        Parameters
        ----------
        step : list
            1 - Imported model
            2 - Imported mapping
            3 - Fetched data
            4 - Selected output folder
            5 - Built model
        '''
        tk.Frame.__init__(self)
        
        # Initialing
        self.c = None
        self.step = []
        self.maxIndexMV = -1
        self.maxIndexDV = -1
        self.maxIndexCV = -1
        self.maxIndexPOV = -1
        self.sConType = tk.StringVar()
        self.sConType.set('SMOC')
        self.bExcEF = tk.BooleanVar(value = True)
        self.OutFolderPath = os.getcwd()
        self.sStatus = tk.StringVar()
        self.sStatus.set('')
        
        # Setting widgets
        self.setup_widgets()
        
    def setup_widgets(self):
        # Check status function
        def checkStatus():
            if 1 in self.step and 2 in self.step:
                self.btnFetch.configure(state = tk.NORMAL)
                if 3 in self.step:
                    self.btnRun.configure(state = tk.NORMAL)
                else:
                    self.btnRun.configure(state = tk.DISABLED)
            else:
                self.btnFetch.configure(state = tk.DISABLED)
                self.btnRun.configure(state = tk.DISABLED)
        
        # Updae status function
        def updateStatus(NewActivity):
            # Check number of line
            text = self.sStatus.get()
            count = text.count('\n')
            
            # Update new status
            if count >= 25:
                self.sStatus.set(text[text.find('\n')+1:] + NewActivity)
            else:
                self.sStatus.set(text + NewActivity)
        
        # Controller type switch function
        def changeState():
            if self.sConType.get().upper() == 'SMOC':
                updateStatus('SMOC is selected. All settings have been reseted.\n')
            else:
                updateStatus('PACE is selected. All settings have been reseted.\n')
            
            # Reset input values
            self.step = []
            self.ModelPath = None
            self.txtModelPath.configure(state = tk.NORMAL)
            self.txtModelPath.delete(0, tk.END)
            self.txtModelPath.insert(0, 'Please select model file')
            self.txtModelPath.configure(state = tk.DISABLED)
            self.txtMappingPath.configure(state = tk.NORMAL)
            self.txtMappingPath.delete(0, tk.END)
            self.txtMappingPath.insert(0, 'Please select tag mapping file')
            self.txtMappingPath.configure(state = tk.DISABLED)
            checkStatus()
        
        # Controller type switch
        self.switch = ttk.Checkbutton(
            self, style = 'Switch.TCheckbutton', textvariable = self.sConType,
            variable = self.sConType, offvalue = 'SMOC', onvalue = 'PACE', command = changeState
        )
        self.switch.grid(row = 0, column = 0, padx = 5, pady = 10, sticky = 'nw')
        
        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row = 1, column = 0, padx = 5, sticky = 'nsew')
        
        ####################################
        ########## Page 1 - Start ##########
        ####################################
        
        # Load images
        ImgUp = tk.PhotoImage(file = 'image/up_s.png')
        ImgDn = tk.PhotoImage(file = 'image/down_s.png')
        
        # Browse file function
        def browseFile(types):
            if types == 'Model':
                # Check controller type
                if self.sConType.get().upper() == 'SMOC':
                    filetypes = [('Hyper Text Markup Language files', '*.htm*')]
                else:
                    filetypes = [('eXtensible Markup Language files', '*.xml')]
                
                # Browse file
                self.ModelPath = filedialog.askopenfilename(
                    initialdir = os.getcwd(), title = 'Select Model File', filetypes = filetypes
                )
                
                # Update text box
                if self.ModelPath != '':
                    self.txtModelPath.configure(state = tk.NORMAL)
                    self.txtModelPath.delete(0, tk.END)
                    self.txtModelPath.insert(0, self.ModelPath)
                    self.txtModelPath.configure(state = tk.DISABLED)
                    self.step.append(1)
                    updateStatus(f'<{self.ModelPath[self.ModelPath.rfind("/")+1:]}> has been opened.\n')
                    checkStatus()
                    
            elif types == 'Mapping':
                # Browse file
                filetypes = [('Excel Workbook files', '*.xlsx'), ('Excel 97-2003 Workbook files', '*.xls')]
                self.MappingPath = filedialog.askopenfilename(
                    initialdir = os.getcwd(), title = 'Select Mapping File', filetypes = filetypes
                )
                
                # Update text box
                if self.MappingPath != '':
                    self.txtMappingPath.configure(state = tk.NORMAL)
                    self.txtMappingPath.delete(0, tk.END)
                    self.txtMappingPath.insert(0, self.MappingPath)
                    self.txtMappingPath.configure(state = tk.DISABLED)
                    self.step.append(2)
                    updateStatus(f'<{self.MappingPath[self.MappingPath.rfind("/")+1:]}> has been opened.\n')
                    checkStatus()
                    
            elif types == 'Output Folder':
                # Browse folder
                self.OutFolderPath = filedialog.askdirectory(
                    initialdir = os.getcwd(), title = 'Select Output Folder'
                )
                # Update text box
                if self.OutFolderPath != '':
                    self.txtOutFolderPath.configure(state = tk.NORMAL)
                    self.txtOutFolderPath.delete(0, tk.END)
                    self.txtOutFolderPath.insert(0, self.OutFolderPath)
                    self.txtOutFolderPath.configure(state = tk.DISABLED)
                    self.step.append(4)
                    updateStatus(f'<{self.OutFolderPath[self.OutFolderPath.rfind("/")+1:]}> has been selected as Output Folder.\n')
                    
        # Fetch data function
        def fetchVariable():
            self.step.append(3)
            checkStatus()
            updateStatus('Variable lists are fetched successfully.\n')
            
            # Create controller for first running
            if self.c is None:
                if self.sConType.get().upper() == 'SMOC':
                    self.c = Controller('SMOC', self.ModelPath, self.MappingPath, excEF = self.bExcEF.get())
                    self.ModelPath_o = self.ModelPath
                    self.MappingPath_o = self.MappingPath
                    self.bExcEF_o = self.bExcEF.get()
                else:
                    self.c = Controller('PACE', self.ModelPath, self.MappingPath, self.bExcEF.get())
                    self.ModelPath_o = self.ModelPath
                    self.MappingPath_o = self.MappingPath
                    self.bExcEF_o = self.bExcEF.get()
            
            # Create controller for next running
            if self.sConType.get() == 'SMOC':
                if self.ModelPath != self.ModelPath_o or self.MappingPath != self.MappingPath_o or self.bExcEF.get() != self.bExcEF_o:
                    self.c = Controller('SMOC', self.ModelPath, self.MappingPath, excEF = self.bExcEF.get())
                    self.ModelPath_o = self.ModelPath
                    self.MappingPath_o = self.MappingPath
                    self.bExcEF_o = self.bExcEF.get()
            else:
                if (self.ModelPath != self.ModelPath_o or self.MappingPath != self.MappingPath_o or self.bExcEF.get() != self.bExcEF_o):
                    self.c = Controller('PACE', self.ModelPath, self.InVarPath, self.OutVarPath, self.bExcEF.get())
                    self.ModelPath_o = self.ModelPath
                    self.MappingPath_o = self.MappingPath
                    self.bExcEF_o = self.bExcEF.get()
            
            # Clear existing data in Preprocessing table
            self.txtController.delete(0, tk.END)
            self.tblMV.delete(*self.tblMV.get_children())
            self.tblDV.delete(*self.tblDV.get_children())
            self.tblCV.delete(*self.tblCV.get_children())
            self.tblPOV.delete(*self.tblPOV.get_children())
            
            # Update data to Preprocessing table
            fetchMapping()
            self.txtController.insert(0, self.ModelPath[self.ModelPath.rfind('/') + 1 : self.ModelPath.rfind('.')])
            for i, mv in self.c.MVs.items():
                self.tblMV.insert(parent = '', index = tk.END, iid = i, text = '', values = (i + 1, mv))
            for i, dv in self.c.DVs.items():
                self.tblDV.insert(parent = '', index = tk.END, iid = i, text = '', values = (i + 1, dv))
            for i, cv in self.c.CVs.items():
                self.tblCV.insert(parent = '', index = tk.END, iid = i, text = '', values = (i + 1, cv))
            for i, pov in self.c.POVs.items():
                self.tblPOV.insert(parent = '', index = tk.END, iid = i, text = '', values = (i + 1, pov))
            
        # Event on select MV
        def onSelectedMV(event):
            currentItem = self.tblMV.focus()
            self.selectedIndexMV, self.selectedMV = self.tblMV.item(currentItem)['values']
            self.btnMV2DV.configure(state = tk.NORMAL)
            updateStatus(f'MV{self.selectedIndexMV} <{self.c.MVs[self.selectedIndexMV-1]}> is selected.\n')
            
        # Event on select DV
        def onSelectedDV(event):
            currentItem = self.tblDV.focus()
            self.selectedIndexDV, self.selectedDV = self.tblDV.item(currentItem)['values']
            updateStatus(f'DV{self.selectedIndexDV} <{self.c.DVs[self.selectedIndexDV-1]}> is selected.\n')
            
            # Check move over variables
            if self.selectedDV in self.c.MVo.values():
                self.btnDV2MV.configure(state = tk.NORMAL)
            else:
                self.btnDV2MV.configure(state = tk.DISABLED)
            
            if self.selectedDV in self.c.POVo.values():
                self.btnDV2POV.configure(state = tk.NORMAL)
            else:
                self.btnDV2POV.configure(state = tk.DISABLED)
        
        # Event on select CV
        def onSelectedCV(event):
            currentItem = self.tblCV.focus()
            self.selectedIndexCV, self.selectedCV = self.tblCV.item(currentItem)['values']
            updateStatus(f'CV{self.selectedIndexCV} <{self.c.CVs[self.selectedIndexCV-1]}> is selected.\n')
            self.btnCV2POV.configure(state = tk.NORMAL)
        
        # Event on select POV
        def onSelectedPOV(event):
            currentItem = self.tblPOV.focus()
            self.selectedIndexPOV, self.selectedPOV = self.tblPOV.item(currentItem)['values']
            updateStatus(f'POV{self.selectedIndexPOV} <{self.c.POVs[self.selectedIndexPOV-1]}> is selected.\n')
            
            # Check move over variables
            if self.selectedPOV not in self.c.CVo.values():
                self.btnPOV2DV.configure(state = tk.NORMAL)
            else:
                self.btnPOV2DV.configure(state = tk.DISABLED)
            
            if self.selectedPOV in self.c.CVo.values():
                self.btnPOV2CV.configure(state = tk.NORMAL)
            else:
                self.btnPOV2CV.configure(state = tk.DISABLED)
        
        # Event move up
        def moveUp(VarType):
            if VarType.upper() == 'MV':
                if self.selectedIndexMV > 1:
                    self.c.moveUp(VarType.upper(), self.selectedIndexMV-1)
                    updateStatus(f'MV{self.selectedIndexMV} <{self.c.MVs[self.selectedIndexMV-1]}> ' + 
                                 f'is moved from MV{self.selectedIndexMV} to MV{self.selectedIndexMV-1}.\n')
                    self.selectedIndexMV -= 1
            if VarType.upper() == 'DV':
                if self.selectedIndexDV > 1:
                    self.c.moveUp(VarType.upper(), self.selectedIndexDV-1)
                    updateStatus(f'DV{self.selectedIndexDV} <{self.c.DVs[self.selectedIndexDV-1]}> ' + 
                                 f'is moved from DV{self.selectedIndexDV} to DV{self.selectedIndexDV-1}.\n')
                    self.selectedIndexDV -= 1
            if VarType.upper() == 'CV':
                if self.selectedIndexCV > 1:
                    self.c.moveUp(VarType.upper(), self.selectedIndexCV-1)
                    updateStatus(f'CV{self.selectedIndexCV} <{self.c.CVs[self.selectedIndexCV-1]}> ' + 
                                 f'is moved from CV{self.selectedIndexCV} to CV{self.selectedIndexCV-1}.\n')
                    self.selectedIndexCV -= 1
            if VarType.upper() == 'POV':
                if self.selectedIndexPOV > 1:
                    self.c.moveUp(VarType.upper(), self.selectedIndexPOV-1)
                    updateStatus(f'POV{self.selectedIndexPOV} <{self.c.POVs[self.selectedIndexPOV-1]}> ' + 
                                 f'is moved from POV{self.selectedIndexPOV} to POV{self.selectedIndexPOV-1}.\n')
                    self.selectedIndexPOV -= 1
            
            fetchVariable()
            fetchMapping()
        
        # Event move down
        def moveDn(VarType):
            if VarType.upper() == 'MV':
                if self.selectedIndexMV-1 != self.maxIndexMV:
                    self.c.moveDn(VarType.upper(), self.selectedIndexMV-1)
                    updateStatus(f'MV{self.selectedIndexMV} <{self.c.MVs[self.selectedIndexMV-1]}> ' + 
                                 f'is moved from MV{self.selectedIndexMV} to MV{self.selectedIndexMV+1}.\n')
                    self.selectedIndexMV += 1
            if VarType.upper() == 'DV':
                if self.selectedIndexDV-1 != self.maxIndexDV:
                    self.c.moveDn(VarType.upper(), self.selectedIndexDV-1)
                    updateStatus(f'DV{self.selectedIndexDV} <{self.c.DVs[self.selectedIndexDV-1]}> ' + 
                                 f'is moved from DV{self.selectedIndexDV} to DV{self.selectedIndexDV+1}.\n')
                    self.selectedIndexDV += 1
            if VarType.upper() == 'CV':
                if self.selectedIndexCV-1 != self.maxIndexCV:
                    self.c.moveDn(VarType.upper(), self.selectedIndexCV-1)
                    updateStatus(f'CV{self.selectedIndexCV} <{self.c.CVs[self.selectedIndexCV-1]}> ' + 
                                 f'is moved from CV{self.selectedIndexCV} to CV{self.selectedIndexCV+1}.\n')
                    self.selectedIndexCV += 1
            if VarType.upper() == 'POV':
                if self.selectedIndexPOV-1 != self.maxIndexPOV:
                    self.c.moveDn(VarType.upper(), self.selectedIndexPOV-1)
                    updateStatus(f'POV{self.selectedIndexPOV} <{self.c.POVs[self.selectedIndexPOV-1]}> ' + 
                                 f'is moved from POV{self.selectedIndexPOV} to POV{self.selectedIndexPOV+1}.\n')
                    self.selectedIndexPOV += 1
            
            fetchVariable()
            fetchMapping()
        
        # Event move from MV to DV
        def moveMV2DV():
            if self.c.DVs != {}:
                self.maxIndexDV = max(self.c.DVs.keys())
            updateStatus(f'MV{self.selectedIndexMV} <{self.c.MVs[self.selectedIndexMV-1]}> is moved to DV.\n')
            self.c.moveMV2DV(self.selectedIndexMV-1, self.maxIndexDV+1)
            self.selectedIndexMV = 0
            self.btnMV2DV.configure(state = tk.DISABLED)
            self.c.resetIndex('MV')
            fetchVariable()
            fetchMapping()
            
        # Event move from DV to MV
        def moveDV2MV():
            if self.c.MVs != {}:
                self.maxIndexMV = max(self.c.MVs.keys())
            updateStatus(f'DV{self.selectedIndexDV} <{self.c.DVs[self.selectedIndexDV-1]}> is moved to MV.\n')
            self.c.moveDV2MV(self.selectedIndexDV-1, self.maxIndexMV+1)
            self.selectedIndexDV = 0
            self.btnDV2MV.configure(state = tk.DISABLED)
            self.c.resetIndex('DV')
            fetchVariable()
            fetchMapping()
            
        # Event move from DV to POV
        def moveDV2POV():
            if self.c.POVs != {}:
                self.maxIndexPOV = max(self.c.POVs.keys())
            updateStatus(f'DV{self.selectedIndexDV} <{self.c.DVs[self.selectedIndexDV-1]}> is moved to POV.\n')
            self.c.moveDV2POV(self.selectedIndexDV-1, self.maxIndexPOV+1)
            self.selectedIndexDV = 0
            self.btnDV2POV.configure(state = tk.DISABLED)
            self.c.resetIndex('DV')
            fetchVariable()
            fetchMapping()
            
        # Event move from CV to POV
        def moveCV2POV():
            if self.c.POVs != {}:
                self.maxIndexPOV = max(self.c.POVs.keys())
            updateStatus(f'CV{self.selectedIndexCV} <{self.c.CVs[self.selectedIndexCV-1]}> is moved to POV.\n')
            self.c.moveCV2POV(self.selectedIndexCV-1, self.maxIndexPOV+1)
            self.selectedIndexCV = 0
            self.btnCV2POV.configure(state = tk.DISABLED)
            self.c.resetIndex('CV')
            fetchVariable()
            fetchMapping()
            
        # Event move from POV to DV
        def movePOV2DV():
            if self.c.DVs != {}:
                self.maxIndexDV = max(self.c.DVs.keys())
            updateStatus(f'POV{self.selectedIndexPOV} <{self.c.POVs[self.selectedIndexPOV-1]}> is moved to DV.\n')
            self.c.movePOV2DV(self.selectedIndexPOV-1, self.maxIndexDV+1)
            self.selectedIndexPOV = 0
            self.btnPOV2DV.configure(state = tk.DISABLED)
            self.c.resetIndex('POV')
            fetchVariable()
            fetchMapping()
            
        # Event move from POV to CV
        def movePOV2CV():
            if self.c.CVs != {}:
                self.maxIndexCV = max(self.c.CVs.keys())
            updateStatus(f'POV{self.selectedIndexPOV} <{self.c.POVs[self.selectedIndexPOV-1]}> is moved to CV.\n')
            self.c.movePOV2CV(self.selectedIndexPOV-1, self.maxIndexCV+1)
            self.selectedIndexPOV = 0
            self.btnPOV2CV.configure(state = tk.DISABLED)
            self.c.resetIndex('POV')
            fetchVariable()
            fetchMapping()
            
        # Page 1 tab
        self.page1 = ttk.Frame(self.notebook)
        for index in [0, 1]:
            self.page1.columnconfigure(index = index, weight = 1)
            self.page1.rowconfigure(index = index, weight = 1)
        self.notebook.add(self.page1, text = 'Inputs')
        
        ###################
        ### Order frame ###
        ###################
        
        self.P1Order = ttk.Frame(self.page1)
        self.P1Order.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Controller
        self.lblController = ttk.Label(self.P1Order, text = 'Controller Name :')
        self.lblController.grid(row = 0, column = 1, padx = 8, stick = 'w')
        self.lblController.configure(font = ('Segoe Ui', '12', 'bold'))
        self.txtController = ttk.Entry(self.P1Order, width = 25)
        self.txtController.grid(row = 0, column = 1, pady = 5, stick = 'ne')
        
        ########## IN ##########
        
        # Input frame
        self.P1OrderIn = ttk.Frame(self.P1Order)
        self.P1OrderIn.grid(row = 1, column = 0, stick = 'nsew')
        
        ########## MV ##########
        
        # MV frame
        self.P1MV = ttk.LabelFrame(self.P1OrderIn, text = 'Manipulated Variables')
        self.P1MV.pack(fill = tk.X, padx = 2)
        
        # MV content
        self.P1MV_1 = ttk.Frame(self.P1MV)
        self.P1MV_1.pack(side = tk.LEFT, ipadx = 45)
        
        # MV scrollbar
        self.sclMV = ttk.Scrollbar(self.P1MV_1)
        self.sclMV.pack(side = tk.RIGHT, fill = tk.Y)
        
        # MV table
        self.tblMV = ttk.Treeview(
            self.P1MV_1, selectmode = 'browse', height = 9,
            yscrollcommand = self.sclMV.set, columns = ('No', 'Tag Name')
        )
        self.tblMV.column('#0', width = 0, stretch = tk.NO)
        self.tblMV.column('No', anchor = tk.CENTER, width = 2)
        self.tblMV.column('Tag Name', anchor = tk.CENTER, width = 120)
        self.tblMV.heading('#0', text = '', anchor = tk.CENTER)
        self.tblMV.heading('No', text = 'No', anchor = tk.CENTER)
        self.tblMV.heading('Tag Name', text = 'Tag Name', anchor = tk.CENTER)
        self.tblMV.bind('<ButtonRelease-1>', onSelectedMV)
        self.tblMV.pack(expand = True, padx = 5, pady = 5, fill = tk.BOTH)
        self.sclMV.config(command = self.tblMV.yview)
        
        # MV control
        self.P1MV_2 = ttk.Frame(self.P1MV)
        self.P1MV_2.pack(side = tk.RIGHT)
        
        # MV move up
        self.btnMVUp = ttk.Button(self.P1MV_2, width = 4, image = ImgUp, 
                                  command=lambda: moveUp('MV'))
        self.btnMVUp.image = ImgUp
        self.btnMVUp.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # MV move to DV
        self.btnMV2DV = ttk.Button(self.P1MV_2, width = 4, text = 'DV',
                                   state = tk.DISABLED, command = moveMV2DV)
        self.btnMV2DV.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # MV move down
        self.btnMVDn = ttk.Button(self.P1MV_2, width = 4, image = ImgDn,
                                  command = lambda: moveDn('MV'))
        self.btnMVDn.image = ImgDn
        self.btnMVDn.grid(row = 2, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        ########## DV ##########
        
        # DV frame
        self.P1DV = ttk.LabelFrame(self.P1OrderIn, text = 'Disturbance Variables')
        self.P1DV.pack(fill = tk.X, padx = 2)
        
        # DV content
        self.P1DV_1 = ttk.Frame(self.P1DV)
        self.P1DV_1.pack(side = tk.LEFT, ipadx = 45)
        
        # DV scrollbar
        self.sclDV = ttk.Scrollbar(self.P1DV_1)
        self.sclDV.pack(side = tk.RIGHT, fill = tk.Y)
        
        # DV table
        self.tblDV = ttk.Treeview(
            self.P1DV_1, selectmode = 'browse', height = 9,
            yscrollcommand = self.sclDV.set, columns = ('No', 'Tag Name')
        )
        self.tblDV.column('#0', width = 0, stretch = tk.NO)
        self.tblDV.column('No', anchor = tk.CENTER, width = 2)
        self.tblDV.column('Tag Name', anchor = tk.CENTER, width = 120)
        self.tblDV.heading('#0', text = '', anchor = tk.CENTER)
        self.tblDV.heading('No', text = 'No', anchor = tk.CENTER)
        self.tblDV.heading('Tag Name', text = 'Tag Name', anchor = tk.CENTER)
        self.tblDV.bind('<ButtonRelease-1>', onSelectedDV)
        self.tblDV.pack(expand = True, padx = 5, pady = 5, fill = tk.BOTH)
        self.sclDV.config(command = self.tblDV.yview)
        
        # DV control
        self.P1DV_2 = ttk.Frame(self.P1DV)
        self.P1DV_2.pack(side = tk.RIGHT)
        
        # DV move up
        self.btnDVUp = ttk.Button(self.P1DV_2, width = 3, image = ImgUp, 
                                  command = lambda: moveUp('DV'))
        self.btnDVUp.image = ImgUp
        self.btnDVUp.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # DV move to MV
        self.btnDV2MV = ttk.Button(self.P1DV_2, width = 4, text = 'MV', 
                                   state = tk.DISABLED, command = moveDV2MV)
        self.btnDV2MV.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # DV move to POV
        self.btnDV2POV = ttk.Button(self.P1DV_2, width = 4, text = 'POV', 
                                    state = tk.DISABLED, command = moveDV2POV)
        self.btnDV2POV.grid(row = 2, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # DV move down
        self.btnDVDn = ttk.Button(self.P1DV_2, width = 4, image = ImgDn, 
                                  command = lambda: moveDn('DV'))
        self.btnDVDn.image = ImgDn
        self.btnDVDn.grid(row = 3, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        ########## OUT ##########
        
        # Output frame
        self.P1OrderOut = ttk.Frame(self.P1Order)
        self.P1OrderOut.grid(row = 1, column = 1, stick = 'nsew')
        
        ########## CV ##########
        
        # CV frame
        self.P1CV = ttk.LabelFrame(self.P1OrderOut, text = 'Control Variables')
        self.P1CV.pack(fill = tk.X, padx = 2)
        
        # CV content
        self.P1CV_1 = ttk.Frame(self.P1CV)
        self.P1CV_1.pack(side = tk.LEFT, ipadx = 45)
        
        # CV scrollbar
        self.sclCV = ttk.Scrollbar(self.P1CV_1)
        self.sclCV.pack(side = tk.RIGHT, fill = tk.Y)
        
        # CV table
        self.tblCV = ttk.Treeview(
            self.P1CV_1, selectmode = 'browse', height = 9, 
            yscrollcommand = self.sclCV.set, columns = ('No', 'Tag Name')
        )
        self.tblCV.column('#0', width = 0, stretch = tk.NO)
        self.tblCV.column('No', anchor = tk.CENTER, width = 2)
        self.tblCV.column('Tag Name', anchor = tk.CENTER, width = 120)
        self.tblCV.heading('#0', text = '', anchor = tk.CENTER)
        self.tblCV.heading('No', text = 'No', anchor = tk.CENTER)
        self.tblCV.heading('Tag Name', text = 'Tag Name', anchor = tk.CENTER)
        self.tblCV.bind('<ButtonRelease-1>', onSelectedCV)
        self.tblCV.pack(expand = True, padx = 5, pady = 5, fill = tk.BOTH)
        self.sclCV.config(command = self.tblCV.yview)
        
        # CV control
        self.P1CV_2 = ttk.Frame(self.P1CV)
        self.P1CV_2.pack(side = tk.RIGHT)
        
        # CV move up
        self.btnCVUp = ttk.Button(self.P1CV_2, width = 4, image = ImgUp, 
                                  command = lambda: moveUp('CV'))
        self.btnCVUp.image = ImgUp
        self.btnCVUp.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # CV move to POV
        self.btnCV2POV = ttk.Button(self.P1CV_2, width = 4, text = 'POV',
                                    state = tk.DISABLED, command = moveCV2POV)
        self.btnCV2POV.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # CV move down
        self.btnCVDn = ttk.Button(self.P1CV_2, width = 4, image = ImgDn,
                                  command = lambda: moveDn('CV'))
        self.btnCVDn.image = ImgDn
        self.btnCVDn.grid(row = 2, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        ########## POV ##########
        
        # POV frame
        self.P1POV = ttk.LabelFrame(self.P1OrderOut, text = 'Process Output Variables')
        self.P1POV.pack(fill = tk.X, padx = 2)
        
        # POV content
        self.P1POV_1 = ttk.Frame(self.P1POV)
        self.P1POV_1.pack(side = tk.LEFT, ipadx = 45)
        
        # POV scrollbar
        self.sclPOV = ttk.Scrollbar(self.P1POV_1)
        self.sclPOV.pack(side = tk.RIGHT, fill = tk.Y)
        
        # POV table
        self.tblPOV = ttk.Treeview(
            self.P1POV_1, selectmode = 'browse', height = 9, 
            yscrollcommand = self.sclPOV.set, columns = ('No', 'Tag Name')
        )
        self.tblPOV.column('#0', width = 0, stretch = tk.NO)
        self.tblPOV.column('No', anchor = tk.CENTER, width = 2)
        self.tblPOV.column('Tag Name', anchor = tk.CENTER, width = 120)
        self.tblPOV.heading('#0', text = '', anchor = tk.CENTER)
        self.tblPOV.heading('No', text = 'No', anchor = tk.CENTER)
        self.tblPOV.heading('Tag Name', text = 'Tag Name', anchor = tk.CENTER)
        self.tblPOV.bind('<ButtonRelease-1>', onSelectedPOV)
        self.tblPOV.pack(expand = True, padx = 5, pady = 5, fill = tk.BOTH)
        self.sclPOV.config(command = self.tblPOV.yview)
        
        # POV control
        self.P1POV_2 = ttk.Frame(self.P1POV)
        self.P1POV_2.pack(side = tk.RIGHT)
        
        # POV move up
        self.btnPOVUp = ttk.Button(self.P1POV_2, width = 4, image = ImgUp,
                                   command = lambda: moveUp('POV'))
        self.btnPOVUp.image = ImgUp
        self.btnPOVUp.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # POV move to CV
        self.btnPOV2CV = ttk.Button(self.P1POV_2, width = 4, text = 'CV', 
                                    state = tk.DISABLED, command = movePOV2CV)
        self.btnPOV2CV.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # POV move to DV
        self.btnPOV2DV = ttk.Button(self.P1POV_2, width=4, text='DV', state=tk.DISABLED,
                                    command = movePOV2DV)
        self.btnPOV2DV.grid(row = 2, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        # POV move down
        self.btnPOVDn = ttk.Button(self.P1POV_2, width = 4, image = ImgDn, 
                                   command = lambda: moveDn('POV'))
        self.btnPOVDn.image = ImgDn
        self.btnPOVDn.grid(row = 3, column = 0, padx = 5, pady = 2, sticky = 'ew')
        
        ###################
        ### Input frame ###
        ###################
        
        self.P1Inputs = ttk.Frame(self.page1)
        self.P1Inputs.pack(fill = tk.X)
        
        # Entry model path
        self.txtModelPath = ttk.Entry(self.P1Inputs, width = 72)
        self.txtModelPath.insert(0, 'Please select model file')
        self.txtModelPath.config(state = tk.DISABLED)
        self.txtModelPath.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'new')
        
        # Browse button for model path
        self.btnModelPath = ttk.Button(
            self.P1Inputs, text = 'Browse', command = lambda: browseFile('Model')
        )
        self.btnModelPath.grid(row = 0, column = 1, pady = 2, sticky = 'new')
         
        # Entry mapping path
        self.txtMappingPath = ttk.Entry(self.P1Inputs, width = 72)
        self.txtMappingPath.insert(0, 'Please select tag mapping file')
        self.txtMappingPath.config(state = tk.DISABLED)
        self.txtMappingPath.grid(row = 1, column = 0, padx = 5, pady = 2, sticky = 'new')
        
        # Browse button for mapping path
        self.btnMappingPath = ttk.Button(
            self.P1Inputs, text = 'Browse', command = lambda: browseFile('Mapping')
        )
        self.btnMappingPath.grid(row = 1, column = 1, pady = 2, sticky = 'new')
        
        # Checkbox excluding economic funciton variables
        self.chkExcEF = ttk.Checkbutton(
            self.P1Inputs, text = 'Excluded economic function varaibles', variable = self.bExcEF
        )
        self.chkExcEF.grid(row = 2, column = 0, padx = 2, pady = 2, sticky = 'new')
        
        # Browse button for output variable path
        self.btnFetch = ttk.Button(
            self.P1Inputs, text = 'Fetch Data', state = tk.DISABLED, command = fetchVariable
        )
        self.btnFetch.grid(row = 2, column = 1, pady = 2, sticky = 'new')
        
        # Blank frame
        self.P1Blank = ttk.Frame(self.page1)
        self.P1Blank.pack(fill = tk.BOTH)
        
        ####################################
        ########### Page 1 - End ###########
        ####################################
        
        ####################################
        ########## Page 2 - Start ##########
        ####################################
        
        def onSelectedMapping(event):
            # Get current item
            currentItem = self.tblMapping.focus()
            self.selectedMapping = self.tblMapping.item(currentItem)['values']
            updateStatus(f'Tag Maaping <{self.selectedMapping[0]} - {self.selectedMapping[1]}> is selected.\n')
            
            # Variable
            self.manVariable.configure(state = tk.NORMAL)
            self.manVariable.delete(0, tk.END)
            self.manVariable.insert(0, self.selectedMapping[0])
            self.manVariable.configure(state = tk.DISABLED)
            
            # Enable
            self.manName.configure(state = tk.NORMAL)
            self.manTagName.configure(state = tk.NORMAL)
            self.manMode.configure(state = tk.NORMAL)
            self.manHigh.configure(state = tk.NORMAL)
            self.manLow.configure(state = tk.NORMAL)
            self.manSST.configure(state = tk.NORMAL)
            self.manPredErr.configure(state = tk.NORMAL)
            self.manOptMode.configure(state = tk.NORMAL)
            self.manOptCoeff.configure(state = tk.NORMAL)
            self.manMaxMove.configure(state = tk.NORMAL)
            self.manWindUp.configure(state = tk.NORMAL)
            
            # Name
            self.manName.delete(0, tk.END)
            self.manName.insert(0, self.selectedMapping[1])
            
            # Tag Name
            self.manTagName.delete(0, tk.END)
            self.manTagName.insert(0, self.selectedMapping[2])
            
            # Mode
            self.manMode.delete(0, tk.END)
            self.manMode.insert(0, self.selectedMapping[3])
            
            # High
            self.manHigh.delete(0, tk.END)
            self.manHigh.insert(0, self.selectedMapping[4])
            
            # Low
            self.manLow.delete(0, tk.END)
            self.manLow.insert(0, self.selectedMapping[5])
            
            # SST
            self.manSST.delete(0, tk.END)
            self.manSST.insert(0, self.selectedMapping[6])
            
            # Predict error
            self.manPredErr.delete(0, tk.END)
            self.manPredErr.insert(0, self.selectedMapping[7])
            
            # Optimization mode
            self.manOptMode.delete(0, tk.END)
            self.manOptMode.insert(0, self.selectedMapping[8])
            
            # Optimization coefficient
            self.manOptCoeff.delete(0, tk.END)
            self.manOptCoeff.insert(0, self.selectedMapping[9])
            
            # Max move
            self.manMaxMove.delete(0, tk.END)
            self.manMaxMove.insert(0, self.selectedMapping[10])
            
            # Wind up
            self.manWindUp.delete(0, tk.END)
            self.manWindUp.insert(0, self.selectedMapping[11])
            
            # Disable
            if self.selectedMapping[0] == 'CON':
                self.manName.configure(state = tk.DISABLED)
                self.manTagName.configure(state = tk.DISABLED)
                self.manMode.configure(state = tk.NORMAL)
                self.manHigh.configure(state = tk.DISABLED)
                self.manLow.configure(state = tk.DISABLED)
                self.manSST.configure(state = tk.DISABLED)
                self.manPredErr.configure(state = tk.DISABLED)
                self.manOptMode.configure(state = tk.NORMAL)
                self.manOptCoeff.configure(state = tk.DISABLED)
                self.manMaxMove.configure(state = tk.DISABLED)
                self.manWindUp.configure(state = tk.DISABLED)
            elif 'MV' in self.selectedMapping[0]:
                self.manName.configure(state = tk.NORMAL)
                self.manTagName.configure(state = tk.NORMAL)
                self.manMode.configure(state = tk.NORMAL)
                self.manHigh.configure(state = tk.NORMAL)
                self.manLow.configure(state = tk.NORMAL)
                self.manSST.configure(state = tk.NORMAL)
                self.manPredErr.configure(state = tk.DISABLED)
                self.manOptMode.configure(state = tk.NORMAL)
                self.manOptCoeff.configure(state = tk.NORMAL)
                self.manMaxMove.configure(state = tk.NORMAL)
                self.manWindUp.configure(state = tk.NORMAL)
            elif 'DV' in self.selectedMapping[0]:
                self.manName.configure(state = tk.NORMAL)
                self.manTagName.configure(state = tk.NORMAL)
                self.manMode.configure(state = tk.NORMAL)
                self.manHigh.configure(state = tk.DISABLED)
                self.manLow.configure(state = tk.DISABLED)
                self.manSST.configure(state = tk.DISABLED)
                self.manPredErr.configure(state = tk.DISABLED)
                self.manOptMode.configure(state = tk.DISABLED)
                self.manOptCoeff.configure(state = tk.DISABLED)
                self.manMaxMove.configure(state = tk.DISABLED)
                self.manWindUp.configure(state = tk.DISABLED)
            elif 'CV' in self.selectedMapping[0]:
                self.manName.configure(state = tk.NORMAL)
                self.manTagName.configure(state = tk.NORMAL)
                self.manMode.configure(state = tk.NORMAL)
                self.manHigh.configure(state = tk.NORMAL)
                self.manLow.configure(state = tk.NORMAL)
                self.manSST.configure(state = tk.NORMAL)
                self.manPredErr.configure(state = tk.NORMAL)
                self.manOptMode.configure(state = tk.NORMAL)
                self.manOptCoeff.configure(state = tk.NORMAL)
                self.manMaxMove.configure(state = tk.DISABLED)
                self.manWindUp.configure(state = tk.DISABLED)
        
        def fetchMapping():
           # Clear existing data in Preprocessing table
            self.tblMapping.delete(*self.tblMapping.get_children())
            
            # Update data to Preprocessing table
            self.tblMapping.insert(parent = '', index = tk.END, text = '',
                                   values = ('CON', 'Controller', '', self.c.Mappings.iloc[0, 2], 
                                             '', '', '', '', '', '', '', ''))
            
            for i, mv in self.c.MVs.items():
                self.tblMapping.insert(parent = '', index = tk.END, text = '',
                                       values = (f'MV0{i+1}' if i < 9 else f'MV{i+1}', mv,
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'Tag'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'Mode'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'High'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'Low'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'SST'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'PredErr'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'OptMode'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'OptCoeff'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'MaxMove'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == mv, 'WindUp'].item()))
            
            for i, dv in self.c.DVs.items():
                self.tblMapping.insert(parent = '', index = tk.END, text = '',
                                       values = (f'DV0{i+1}' if i < 9 else f'DV{i+1}', dv,
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'Tag'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'Mode'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'High'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'Low'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'SST'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'PredErr'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'OptMode'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'OptCoeff'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'MaxMove'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == dv, 'WindUp'].item()))
            
            for i, cv in self.c.CVs.items():
                self.tblMapping.insert(parent = '', index = tk.END, text = '',
                                       values = (f'CV0{i+1}' if i < 9 else f'CV{i+1}', cv,
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'Tag'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'Mode'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'High'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'Low'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'SST'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'PredErr'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'OptMode'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'OptCoeff'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'MaxMove'].item(),
                                                 self.c.Mappings.loc[self.c.Mappings['Name'] == cv, 'WindUp'].item()))
        
        def updateMapping():
            updateStatus(f'Tag Maaping <{self.selectedMapping[0]} - {self.selectedMapping[1]}> is updated.\n')
            
            # Update value
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'Tag'] = self.manTagName.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'Mode'] = self.manMode.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'High'] = self.manHigh.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'Low'] = self.manLow.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'SST'] = self.manSST.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'PredErr'] = self.manPredErr.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'OptMode'] = self.manOptMode.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'OptCoeff'] = self.manOptCoeff.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'MaxMove'] = self.manMaxMove.get()
            self.c.Mappings.loc[self.c.Mappings['Name'] == self.manName.get(), 'WindUp'] = self.manWindUp.get()
            fetchMapping()
            
            # Clear value
            self.manVariable.configure(state = tk.NORMAL)
            self.manVariable.delete(0, tk.END)
            self.manVariable.configure(state = tk.DISABLED)
            self.manName.configure(state = tk.NORMAL)
            self.manName.delete(0, tk.END)
            self.manName.configure(state = tk.DISABLED)
            self.manTagName.delete(0, tk.END)
            self.manMode.delete(0, tk.END)
            self.manHigh.delete(0, tk.END)
            self.manLow.delete(0, tk.END)
            self.manSST.delete(0, tk.END)
            self.manPredErr.delete(0, tk.END)
            self.manOptMode.delete(0, tk.END)
            self.manOptCoeff.delete(0, tk.END)
            self.manMaxMove.delete(0, tk.END)
            self.manWindUp.delete(0, tk.END)
            
        # Page 2 tab
        self.page2 = ttk.Frame(self.notebook)
        for index in [0, 1]:
            self.page2.columnconfigure(index = index, weight = 1)
            self.page2.rowconfigure(index = index, weight = 1)
        self.notebook.add(self.page2, text = 'Tag Mapping')
        
        # Mapping frame
        self.frmMapping = ttk.Frame(self.page2)
        self.frmMapping.pack(fill = tk.X)
        
        # Mapping scrollbar
        self.sclMapping = ttk.Scrollbar(self.frmMapping)
        self.sclMapping.pack(side = tk.RIGHT, fill = tk.Y)
        
        # Mapping table
        self.tblMapping = ttk.Treeview(self.frmMapping, selectmode = 'browse', height = 20,
                                       yscrollcommand = self.sclMapping.set)
        self.tblMapping['column'] = ['Variable', 'Name', 'Tag Name', 'Mode', 'High', 'Low', 'SST',
                                     'Pred Err', 'Opt Mode', 'Opt Coeff', 'Max Move', 'Wind Up']
        
        self.tblMapping.heading('#0', text = '', anchor = tk.CENTER)
        for col in self.tblMapping['columns']:
            self.tblMapping.heading(col, text = col, anchor = tk.CENTER)
        
        self.tblMapping.column('#0', width = 0, stretch = tk.NO)
        self.tblMapping.column('Variable', width = 75, anchor = tk.CENTER)
        self.tblMapping.column('Name', width = 150, anchor = tk.CENTER)
        self.tblMapping.column('Tag Name', width = 150, anchor = tk.CENTER)
        self.tblMapping.column('Mode', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('High', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('Low', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('SST', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('Pred Err', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('Opt Mode', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('Opt Coeff', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('Max Move', width = 100, anchor = tk.CENTER)
        self.tblMapping.column('Wind Up', width = 100, anchor = tk.CENTER)
        self.tblMapping.bind('<ButtonRelease-1>', onSelectedMapping)
        self.tblMapping.pack(expand = True, fill = tk.BOTH)
        self.sclMapping.config(command = self.tblMapping.yview)
        
        # Manual entry frame
        self.frmManual = ttk.Frame(self.page2)
        self.frmManual.pack(fill = tk.X, pady = 2)
        
        # Variable manual entry
        self.lblVariable = ttk.Label(self.frmManual, text = 'Variable')
        self.lblVariable.grid(row = 0, column = 0, padx = 1, pady = 2)
        self.manVariable = ttk.Entry(self.frmManual, width = 10, state = tk.DISABLED)
        self.manVariable.grid(row = 1, column = 0, padx = 1)
        
        # Name manual entry
        self.lblName = ttk.Label(self.frmManual, text = 'Name')
        self.lblName.grid(row = 0, column = 1, padx = 1, pady = 2)
        self.manName = ttk.Entry(self.frmManual, width = 20)
        self.manName.grid(row = 1, column = 1, padx = 1)
        
        # Tag name manual entry
        self.lblTagName = ttk.Label(self.frmManual, text = 'Tag Name')
        self.lblTagName.grid(row = 0, column = 2, padx = 1, pady = 2)
        self.manTagName = ttk.Entry(self.frmManual, width = 20)
        self.manTagName.grid(row = 1, column = 2, padx = 1)
        
        # Mode manual entry
        self.lblMode = ttk.Label(self.frmManual, text = 'Mode')
        self.lblMode.grid(row = 0, column = 3, padx = 1, pady = 2)
        self.manMode = ttk.Entry(self.frmManual, width = 12)
        self.manMode.grid(row = 1, column = 3, padx = 1)
        
        # High manual entry
        self.lblHigh = ttk.Label(self.frmManual, text = 'High')
        self.lblHigh.grid(row = 0, column = 4, padx = 1, pady = 2)
        self.manHigh = ttk.Entry(self.frmManual, width = 12)
        self.manHigh.grid(row = 1, column = 4, padx = 1)
        
        # Low manual entry
        self.lblLow = ttk.Label(self.frmManual, text = 'Low')
        self.lblLow.grid(row = 0, column = 5, padx = 1, pady = 2)
        self.manLow = ttk.Entry(self.frmManual, width = 12)
        self.manLow.grid(row = 1, column = 5, padx = 1)
        
        # SST manual entry
        self.lblSST = ttk.Label(self.frmManual, text = 'SST')
        self.lblSST.grid(row = 0, column = 6, padx = 1, pady = 2)
        self.manSST = ttk.Entry(self.frmManual, width = 12)
        self.manSST.grid(row = 1, column = 6, padx = 1)
        
        # Predict error manual entry
        self.lblPredErr = ttk.Label(self.frmManual, text = 'Pred Err')
        self.lblPredErr.grid(row = 0, column = 7, padx = 1, pady = 2)
        self.manPredErr = ttk.Entry(self.frmManual, width = 12)
        self.manPredErr.grid(row = 1, column = 7, padx = 1)
        
        # Optimization mode manual entry
        self.lblOptMode = ttk.Label(self.frmManual, text = 'Opt Mode')
        self.lblOptMode.grid(row = 0, column = 8, padx = 1, pady = 2)
        self.manOptMode = ttk.Entry(self.frmManual, width = 12)
        self.manOptMode.grid(row = 1, column = 8, padx = 1)
        
        # Optimization coefficient manual entry
        self.lblOptCoeff = ttk.Label(self.frmManual, text = 'Opt Coeff')
        self.lblOptCoeff.grid(row = 0, column = 9, padx = 1, pady = 2)
        self.manOptCoeff = ttk.Entry(self.frmManual, width = 12)
        self.manOptCoeff.grid(row = 1, column = 9, padx = 1)
        
        # Max move manual entry
        self.lblMaxMove = ttk.Label(self.frmManual, text = 'Max Move')
        self.lblMaxMove.grid(row = 0, column = 10, padx = 1, pady = 2)
        self.manMaxMove = ttk.Entry(self.frmManual, width = 12)
        self.manMaxMove.grid(row = 1, column = 10, padx = 1)
        
        # Wind up manual entry
        self.lblWindUp = ttk.Label(self.frmManual, text = 'Wind Up')
        self.lblWindUp.grid(row = 0, column = 11, padx = 1, pady = 2)
        self.manWindUp = ttk.Entry(self.frmManual, width = 12)
        self.manWindUp.grid(row = 1, column = 11, padx = 1)
        
        # Update manual entry
        self.btnManual = ttk.Button(self.frmManual, text = 'Update', command = updateMapping)
        self.btnManual.grid(row = 2, column = 11, pady = 5)
        
        ####################################
        ########### Page 2 - End ###########
        ####################################
        
        ####################################
        ########## Page 3 - Start ##########
        ####################################
        
        def inverseLaplace(Model):
            # Specify settling time
            if Model[5][0] == 0:
                if len(Model[5]) == 3:
                    ts = (6 / Model[5][2]) + Model[3]
                else:
                    ts = (12 * Model[5][-2] / Model[5][-1]) + Model[3]
            else:
                if len(Model[5]) == 2:
                    ts = (6 / Model[5][1]) + Model[3]
                else:
                    ts = (12 * Model[5][-2] / Model[5][-1]) + Model[3]
            
            # Create time
            nt = 1000
            t  = np.linspace(0, ts, nt)
            
            # Build transfer function
            num = Model[4]
            den = Model[5]
            H = Model[2]*control.tf(num, den)
            
            # Convert to time series
            (t, y) = control.step_response(H, t)
            
            # Apply time delay
            if Model[3] > 0:
                t = [0] + [time + Model[3] for time in t][:-1]
                y = [0] + list(y)[:-1]
            
            return list(t), list(y)
        
        def preprocessFinalModel():
            # Create column and row
            self.VisColumns = list(self.c.MVs.values()) + list(self.c.DVs.values())
            self.VisRows = list(self.c.CVs.values())
            
            # Create model and data
            self.VisModel = np.zeros((len(self.VisRows), len(self.VisColumns), 4, 10))
            self.VisData = np.zeros((len(self.VisRows), len(self.VisColumns), 2, 1000))
            for fm in self.c.FinalModel:
                for i, col in enumerate(self.VisColumns):
                    for o, row in enumerate(self.VisRows):
                        if fm[0] == col and fm[1] == row:
                            # Visualization Model
                            lNum = len(fm[4])
                            lDen = len(fm[5])
                            self.VisModel[o][i][0][0] = fm[2]  # Gain
                            self.VisModel[o][i][1][0] = fm[3]  # Delay
                            self.VisModel[o][i][2][:lNum] = fm[4]  # Numerator
                            self.VisModel[o][i][3][:lDen] = fm[5]  # Denominator
                            
                            # Visualization Data
                            t, y = inverseLaplace(fm)
                            self.VisData[o][i][0][:] = t
                            self.VisData[o][i][1][:] = y
        
        def createMap():
            # Replace name function
            def replaceName(name):
                return name.upper().replace('-', '').replace('.', '_')
            
            # Initializing
            self.TagMapping = []
            
            # Controller
            for con in self.dataCON:
                if (con[3] != '') and (con[3] != '0') and (not con[3][1:].isnumeric()):
                    self.TagMapping.append([con[3], self.txtController.get() + '.MODE'])
                if (con[8] != '') and (con[8] != '0') and (not con[8][1:].isnumeric()):
                    self.TagMapping.append([con[8], self.txtController.get() + '.OPT'])
            
            # MV
            for mv in self.dataMV:
                if (mv[2] != '') and (mv[2] != '0') and (not mv[2][1:].isnumeric()):
                    self.TagMapping.append([mv[2], replaceName(mv[1])])
                if (mv[3] != '') and (mv[3] != '0') and (not mv[3][1:].isnumeric()):
                    self.TagMapping.append([mv[3], replaceName(mv[1]) + '.MODE'])
                if (mv[4] != '') and (mv[4] != '0') and (not mv[4][1:].isnumeric()):
                    self.TagMapping.append([mv[4], replaceName(mv[1]) + '.HIGH'])
                if (mv[5] != '') and (mv[5] != '0') and (not mv[5][1:].isnumeric()):
                    self.TagMapping.append([mv[5], replaceName(mv[1]) + '.LOW'])
                if (mv[6] != '') and (mv[6] != '0') and (not mv[6][1:].isnumeric()):
                    self.TagMapping.append([mv[6], replaceName(mv[1]) + '.SST'])
                if (mv[8] != '') and (mv[8] != '0') and (not mv[8][1:].isnumeric()):
                    self.TagMapping.append([mv[8], replaceName(mv[1]) + '.OPTMODE'])
                if (mv[9] != '') and (mv[9] != '0') and (not mv[9][1:].isnumeric()):
                    self.TagMapping.append([mv[9], replaceName(mv[1]) + '.OPTCOEFF'])
                if (mv[10] != '') and (mv[10] != '0') and (not mv[10][1:].isnumeric()):
                    self.TagMapping.append([mv[10], replaceName(mv[1]) + '.MAXUP'])
                    self.TagMapping.append([mv[10], replaceName(mv[1]) + '.MAXDN'])
                if (mv[11] != '') and (mv[11] != '0') and (not mv[11][1:].isnumeric()):
                    self.TagMapping.append([mv[11], replaceName(mv[1]) + '.WNUP'])
        
            # DV
            for dv in self.dataDV:
                if (dv[2] != '') and (dv[2] != '0') and (not dv[2][1:].isnumeric()):
                    self.TagMapping.append([dv[2], replaceName(dv[1])])
                if (dv[3] != '') and (dv[3] != '0') and (not dv[3][1:].isnumeric()):
                    self.TagMapping.append([dv[3], replaceName(dv[1]) + '.MODE'])
        
            # CV
            for cv in self.dataCV:
                if (cv[2] != '') and (cv[2] != '0') and (not cv[2][1:].isnumeric()):
                    self.TagMapping.append([cv[2], replaceName(cv[1])])
                if (cv[3] != '') and (cv[3] != '0') and (not cv[3][1:].isnumeric()):
                    self.TagMapping.append([cv[3], replaceName(cv[1]) + '.MODE'])
                if (cv[4] != '') and (cv[4] != '0') and (not cv[4][1:].isnumeric()):
                    self.TagMapping.append([cv[4], replaceName(cv[1]) + '.HIGH'])
                if (cv[5] != '') and (cv[5] != '0') and (not cv[5][1:].isnumeric()):
                    self.TagMapping.append([cv[5], replaceName(cv[1]) + '.LOW'])
                if (cv[6] != '') and (cv[6] != '0') and (not cv[6][1:].isnumeric()):
                    self.TagMapping.append([cv[6], replaceName(cv[1]) + '.SST'])
                if (cv[7] != '') and (cv[7] != '0') and (not cv[7][1:].isnumeric()):
                    self.TagMapping.append([cv[7], replaceName(cv[1]) + '.PREDERR'])
                if (cv[8] != '') and (cv[8] != '0') and (not cv[8][1:].isnumeric()):
                    self.TagMapping.append([cv[8], replaceName(cv[1]) + '.OPTMODE'])
                if (cv[9] != '') and (cv[9] != '0') and (not cv[9][1:].isnumeric()):
                    self.TagMapping.append([cv[9], replaceName(cv[1]) + '.OPTCOEFF'])
        
            self.TagMapping = pd.DataFrame(self.TagMapping)
            self.TagMapping.to_csv(f'{self.OutFolderPath}/{self.txtController.get()}_TagMap.csv', header = False, index = False)
        
        def createXMLConfig():
            # Indent function
            def appendXMLTree(parent, tag, extra, close):
                # Add parent hierarchy
                if parent is None and close != 'close':
                    if tag not in self.xmlparent.keys():
                        self.xmlparent[tag] = 0
                    
                    self.xmlcontent = '<?xml version="1.0" encoding="utf-8"?>\n'
                else:
                    # Add parent hierarchy
                    if tag not in self.xmlparent.keys():
                        self.xmlparent[tag] = self.xmlparent[parent]+1
                    
                    self.xmlcontent += '\t'*self.xmlparent[tag]
                
                # Add content
                if close == 'close':
                    self.xmlcontent += f'</{tag}>\n'
                    self.xmlparent.pop(tag, None)
                elif close == 'self-close':
                    if extra is None:
                        self.xmlcontent += f'<{tag} />\n'
                    else:
                        self.xmlcontent += f'<{tag}'
                        for k, v in extra.items():
                            self.xmlcontent += f' {k}="{v}"'
                        self.xmlcontent += ' />\n'
                    self.xmlparent.pop(tag, None)
                elif close == 'open':
                    if extra is None:
                        self.xmlcontent += f'<{tag}>\n'
                    else:
                        self.xmlcontent += f'<{tag}'
                        for k, v in extra.items():
                            self.xmlcontent += f' {k}="{v}"'
                        self.xmlcontent += '>\n'
            
            # Check source data
            def checkSource(vartype, data):
                # Count number of tags
                blank, constant, tag = 0, 0, 0
                for x in data:
                    if x == '' or x == '0':
                        blank += 1
                    elif x[1:].isnumeric():
                        constant += 1
                    elif self.TagMapping[0].str.contains(str(x)).any():
                        tag += 1
                
                if ((vartype == 'MV' and tag == len(self.dataMV)) or 
                    (vartype == 'DV' and tag == len(self.dataDV)) or 
                    (vartype == 'CV' and tag == len(self.dataCV))):
                    return 'External'
                elif tag > 0 and constant > 0:
                    return 'ExternalConfig'
                elif ((vartype == 'MV' and blank == len(self.dataMV)) or 
                      (vartype == 'DV' and blank == len(self.dataDV)) or 
                      (vartype == 'CV' and blank == len(self.dataCV))):
                    return 'NotUsed'
            
            # Replace name function
            def replaceName(name):
                return name.upper().replace('-', '').replace('.', '_')
            
            self.xmlparent = {}
            appendXMLTree(None, 'Configuration', {'xmlns':'http://schemas.matrikon.com/2010/08/CPM/GenericAPC.xsd'}, 'open')
            
            dictMapping = {0:'Variable', 1:'Name', 2:'TagName', 3:'Mode', 4:'High', 5:'Low', 6:'SST', 
                           7:'PredErr', 8:'OptMode', 9:'OptCoeff', 10:'MaxMove', 11:'WindUp'}
            
            # Controller
            appendXMLTree('Configuration', 'Controller', {'Name':self.txtController.get()}, 'open')
            appendXMLTree('Controller', 'ExecutionInterval', {'Value':'60'}, 'self-close')
            appendXMLTree('Controller', 'HP', {'Value':'180'}, 'self-close')
            appendXMLTree('Controller', 'Mode', {'Source':'External'}, 'self-close')
            appendXMLTree('Controller', 'OptMode', {'Source':'NotUsed'}, 'self-close')
            appendXMLTree('Controller', 'OptCost', {'Source':'NotUsed'}, 'self-close')
            appendXMLTree('Configuration', 'Controller', None, 'close')
            
            # MV
            appendXMLTree('Configuration', 'MVs', None, 'open')
            appendXMLTree('MVs', 'Defaults', None, 'open')
            appendXMLTree('Defaults', 'Mode', {'Source':checkSource('MV', self.dataMV[:, 3])}, 'self-close')
            appendXMLTree('Defaults', 'High', {'Source':checkSource('MV', self.dataMV[:, 4])}, 'self-close')
            appendXMLTree('Defaults', 'Low', {'Source':checkSource('MV', self.dataMV[:, 5])}, 'self-close')
            appendXMLTree('Defaults', 'HEnb', {'Source':'Config', 'Value':'1'}, 'self-close')
            appendXMLTree('Defaults', 'LEnb', {'Source':'Config', 'Value':'1'}, 'self-close')
            appendXMLTree('Defaults', 'MaxUp', {'Source':checkSource('MV', self.dataMV[:, 10])}, 'self-close')
            appendXMLTree('Defaults', 'MaxDown', {'Source':checkSource('MV', self.dataMV[:, 10])}, 'self-close')
            appendXMLTree('Defaults', 'MoveEnb', {'Source':'Config', 'Value':'1'}, 'self-close')
            appendXMLTree('Defaults', 'OptMode', {'Source':checkSource('MV', self.dataMV[:, 8])}, 'self-close')
            appendXMLTree('Defaults', 'OptCoeff', {'Source':checkSource('MV', self.dataMV[:, 9])}, 'self-close')
            appendXMLTree('Defaults', 'WindUp', {'Source':checkSource('MV', self.dataMV[:, 11])}, 'self-close')
            appendXMLTree('Defaults', 'SST', {'Source':checkSource('MV', self.dataMV[:, 6])}, 'self-close')
            appendXMLTree('Defaults', 'SP', {'Source':'NotUsed'}, 'self-close')
            appendXMLTree('MVs', 'Defaults', None, 'close')
            
            for i, mv in enumerate(self.c.MVs.values()):
                mvconstant = []
                for j, x in enumerate(self.dataMV[i]):
                    if x[1:].isnumeric():
                        mvconstant.append((j, x))
                
                if mvconstant == []:
                    appendXMLTree('MVs', 'MV', {'Name':replaceName(mv)}, 'self-close')
                else:
                    appendXMLTree('MVs', 'MV', {'Name':replaceName(mv)}, 'open')
                    for j, x in mvconstant:
                        appendXMLTree('MV', dictMapping[j], {'Value':x}, 'self-close')
                    appendXMLTree('MVs', 'MV', None, 'close')
            
            appendXMLTree('Configuration', 'MVs', None, 'close')
            
            # DV
            appendXMLTree('Configuration', 'DVs', None, 'open')
            appendXMLTree('DVs', 'Defaults', None, 'open')
            appendXMLTree('Defaults', 'Mode', {'Source':checkSource('DV', self.dataDV[:, 3])}, 'self-close')
            appendXMLTree('DVs', 'Defaults', None, 'close')
            
            for i, dv in enumerate(self.c.DVs.values()):
                dvconstant = []
                for j, x in enumerate(self.dataDV[i]):
                    if x[1:].isnumeric():
                        dvconstant.append((j, x))
                
                if dvconstant == []:
                    appendXMLTree('DVs', 'DV', {'Name':replaceName(dv)}, 'self-close')
                else:
                    appendXMLTree('DVs', 'DV', {'Name':replaceName(dv)}, 'open')
                    for j, x in dvconstant:
                        appendXMLTree('DV', dictMapping[j], {'Value':x}, 'self-close')
                    appendXMLTree('DVs', 'DV', None, 'close')
            
            appendXMLTree('Configuration', 'DVs', None, 'close')
            
            # CV
            appendXMLTree('Configuration', 'CVs', None, 'open')
            appendXMLTree('CVs', 'Defaults', None, 'open')
            appendXMLTree('Defaults', 'Mode', {'Source':checkSource('CV', self.dataCV[:, 3])}, 'self-close')
            appendXMLTree('Defaults', 'High', {'Source':checkSource('CV', self.dataCV[:, 4])}, 'self-close')
            appendXMLTree('Defaults', 'Low', {'Source':checkSource('CV', self.dataCV[:, 5])}, 'self-close')
            appendXMLTree('Defaults', 'HEnb', {'Source':'Config', 'Value':'1'}, 'self-close')
            appendXMLTree('Defaults', 'LEnb', {'Source':'Config', 'Value':'1'}, 'self-close')
            appendXMLTree('Defaults', 'OptMode', {'Source':checkSource('CV', self.dataCV[:, 8])}, 'self-close')
            appendXMLTree('Defaults', 'OptCoeff', {'Source':checkSource('CV', self.dataCV[:, 9])}, 'self-close')
            appendXMLTree('Defaults', 'PredErr', {'Source':'ExternalPred'}, 'self-close')
            appendXMLTree('Defaults', 'SST', {'Source':checkSource('CV', self.dataCV[:, 7])}, 'self-close')
            appendXMLTree('Defaults', 'SP', {'Source':'NotUsed'}, 'self-close')
            appendXMLTree('MVs', 'Defaults', None, 'close')
            
            for i, cv in enumerate(self.c.CVs.values()):
                cvconstant = []
                for j, x in enumerate(self.dataCV[i]):
                    if x[1:].isnumeric():
                        cvconstant.append((j, x))
                
                if cvconstant == []:
                    appendXMLTree('CVs', 'CV', {'Name':replaceName(cv)}, 'self-close')
                else:
                    appendXMLTree('CVs', 'CV', {'Name':replaceName(cv)}, 'open')
                    for j, x in cvconstant:
                        appendXMLTree('CV', dictMapping[j], {'Value':x}, 'self-close')
                    appendXMLTree('CVs', 'CV', None, 'close')
            
            appendXMLTree('Configuration', 'CVs', None, 'close')
            
            appendXMLTree(None, 'Configuration', None, 'close')
            
            with open(f'{self.OutFolderPath}/{self.txtController.get()}_Configuration.xml', 'w') as f:
                f.write(self.xmlcontent)
        
        def createLO2():
            # Replace name function
            def replaceName(name):
                return name.upper().replace('-', '').replace('.', '_')
            
            # Check max order
            self.maxOrder = 0
            for mdl in self.c.FinalModel:
                if self.maxOrder < len(mdl[5]):
                    self.maxOrder = len(mdl[5])
            
            # Header
            text = 'Normal format: G(s) = Gain*[('
            for o in range(self.maxOrder-2, 0, -1):
                if o == 1:
                    text += 'B1*s+'
                else:
                    text += f'B{o}*s^{o}+'
            text += 'B0)/('
            for o in range(self.maxOrder-1, 0, -1):
                if o == 1:
                    text += 'A1*s+'
                else:
                    text += f'A{o}*s^{o}+'
            text += 'A0)]*exp(-Delay*s)\n\n'
            
            # Detail each MV-CV
            for i, mv in self.c.MVs.items():
                for j, cv in self.c.CVs.items():
                    for mdl in self.c.FinalModel:
                        if mdl[0] == mv and mdl[1] == cv:
                            text += f'MV{i+1}: {replaceName(mv)}, CV{j+1}: {replaceName(cv)}\n'
                            text += f'Gain = {round(mdl[2], 10)}, \t'
                            for o in range(self.maxOrder-2, -1, -1):
                                try:
                                    text += f'B{o} = {round(mdl[4][o], 10)}, \t'
                                except:
                                    text += f'B{o} = 0, \t'
                            for o in range(self.maxOrder-1, -1, -1):
                                try:
                                    text += f'A{o} = {round(mdl[5][o], 10)}, \t'
                                except:
                                    text += f'A{o} = 0, \t'
                            text += f'Delay = {round(mdl[3], 10)}\n\n'
            
            # Write LO2 model file
            with open(f'{self.OutFolderPath}/{self.txtController.get()}_Model.LO2', 'w') as f:
                f.write(text[:-2])
        
        def plotFinalModel():
            # Create figure
            fig, axs = plt.subplots(len(self.VisRows), len(self.VisColumns), figsize = (2.5*len(self.VisColumns), 2.5*len(self.VisRows)))
            for r, row in enumerate(self.VisRows):
                for c, col in enumerate(self.VisColumns):
                    # Plot graph
                    axs[r][c].plot(self.VisData[r][c][0], self.VisData[r][c][1])
                    
                    # Put variable label
                    if r == 0:
                        axs[r][c].set_xlabel(col)
                        axs[r][c].xaxis.set_label_position('top') 
                    if c == 0:
                        axs[r][c].set_ylabel(row)
            
            # Save picture to output folder
            plt.savefig(self.OutFolderPath + f'/{self.txtController.get()}_Model.png')
        
        def plotIndividualModel(event):
            # Get index
            c = [c for c, col in enumerate(self.VisColumns) if col == self.cmbX.get()][0]
            r = [r for r, row in enumerate(self.VisRows) if row == self.cmbY.get()][0]
            
            # Plot graph
            ax.clear()
            ax.plot(self.VisData[r][c][0], self.VisData[r][c][1])
            ax.set_title(f'Step Response: {self.cmbX.get()} - {self.cmbY.get()}')
            fig_canvas.draw()
            
            # Update status
            updateStatus(f'Step Response: {self.cmbX.get()} vs {self.cmbY.get()} ' +
                             'is displayed.\n')
            
        def run():
            # Build model
            self.c.buildModel()
            preprocessFinalModel()
            
            self.dataMapping = np.array([self.tblMapping.item(child)['values'] for child in self.tblMapping.get_children()])
            self.dataCON = self.dataMapping[self.dataMapping[:, 0] == 'CON']
            self.dataMV = self.dataMapping[['MV' in var for var in self.dataMapping[:, 0]]]
            self.dataDV = self.dataMapping[['DV' in var for var in self.dataMapping[:, 0]]]
            self.dataCV = self.dataMapping[['CV' in var for var in self.dataMapping[:, 0]]]
            
            # Generate map file
            createMap()
            updateStatus(f'<{self.txtController.get()}_TagMap.csv> file has been saved successfully.\n')
            
            # Generate xml file
            createXMLConfig()
            updateStatus(f'<{self.txtController.get()}_Configuration.xml> file has been saved successfully.\n')
            
            # Generate LO2 file
            createLO2()
            updateStatus(f'<{self.txtController.get()}_Model.LO2> file has been saved successfully.\n')
            
            # Plot graph
            plotFinalModel()
            updateStatus(f'<{self.txtController.get()}_Model.png> file has been saved successfully.\n')
            
            # Append display variable
            self.XVis = self.VisColumns
            self.YVis = self.VisRows
            self.cmbX.configure(values = self.XVis)
            self.cmbX.current(0)
            self.cmbY.configure(values = self.YVis)
            self.cmbY.current(0)
            plotIndividualModel(True)
            
        # Page 3 tab
        self.page3 = ttk.Frame(self.notebook)
        for index in [0, 1]:
            self.page3.columnconfigure(index = index, weight = 1)
            self.page3.rowconfigure(index = index, weight = 1)
        self.notebook.add(self.page3, text = 'Outputs')
        
        ###############
        ##### Run #####
        ###############
        
        # Command frame
        self.P3Command = ttk.Frame(self.page3)
        self.P3Command.grid(row = 0, column = 0, sticky = 'nw')
        
        # Entry model path
        self.txtOutFolderPath = ttk.Entry(self.P3Command, width = 60)
        self.txtOutFolderPath.insert(0, 'Please select output folder (default is current folder)')
        self.txtOutFolderPath.configure(state = tk.DISABLED)
        self.txtOutFolderPath.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'new')
        
        # Browse button for model path
        self.btnOutFolderPath = ttk.Button(self.P3Command, text = 'Browse', command = lambda: browseFile('Output Folder'))
        self.btnOutFolderPath.grid(row = 0, column = 1, pady = 2, sticky = 'new')
        
        # Run button
        self.btnRun = ttk.Button(self.P3Command, text='Run', state = tk.DISABLED, command = run)
        self.btnRun.grid(row = 1, column = 1, pady = 2, sticky = 'new')
        
        # Status frame
        self.frmStatus = ttk.LabelFrame(self.P3Command, text = 'Message')
        self.frmStatus.grid(row = 2, column = 0, columnspan = 2, pady = 2, ipady = 225, sticky = 'nsew')
        self.frmStatus.pack_propagate(0)
        
        # Status text
        self.lblStatus = ttk.Label(self.frmStatus, textvariable = self.sStatus)
        self.lblStatus.pack(fill = tk.X)
        
        ###############
        ### Preview ###
        ###############
        
        # Preview frame
        self.P3Preview = ttk.LabelFrame(self.page3, text = 'Variable Relationship')
        self.P3Preview.grid(row = 0, column = 1, padx = 2, pady = 5, sticky = 'ne')
        
        # Control frame
        self.frmControl = ttk.Frame(self.P3Preview)
        self.frmControl.grid(row = 0, column = 0, padx = 5, pady = 2, sticky = 'ne')
        
        # X
        self.lblX = ttk.Label(self.frmControl, text = 'Input Variable')
        self.lblX.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = 'ew')
        self.cmbX = ttk.Combobox(self.frmControl, state = 'readonly')
        self.cmbX.grid(row = 1, column = 0, padx = 5, pady = 5, sticky = 'ew')
        self.cmbX.bind('<<ComboboxSelected>>', plotIndividualModel)
        
        # Y
        self.lblY = ttk.Label(self.frmControl, text = 'Output Variable')
        self.lblY.grid(row = 2, column = 0, padx = 5, pady = 5, sticky = 'ew')
        self.cmbY = ttk.Combobox(self.frmControl, state = 'readonly')
        self.cmbY.grid(row = 3, column = 0, padx = 5, pady = 5, sticky = 'ew')
        self.cmbY.bind('<<ComboboxSelected>>', plotIndividualModel)
        
        # Display frame
        self.frmDisplay = ttk.Frame(self.P3Preview, width = 600, height = 570)
        self.frmDisplay.grid(row = 0, column = 1, padx = 5, pady = 2, sticky = 'nw')
        
        # Plot graph
        fig = plt.Figure(figsize = (6, 5), dpi = 100)
        fig_canvas = FigureCanvasTkAgg(fig, self.frmDisplay)
        fig_canvas.get_tk_widget().pack()
        ax = fig.add_subplot(111)
        
        ####################################
        ########### Page 3 - End ###########
        ####################################

# %% Main program

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Yokogawa APC Conversion for CPM')

    # Set theme
    root.tk.call('source', 'azure.tcl')
    root.tk.call('set_theme', 'light')
    root.iconphoto(False, tk.PhotoImage(master=root, file='image/icon.png'))

    app = App(root)
    app.pack(fill='both', expand=True)

    # Set a minsize for the window, and place it in the middle
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.maxsize(root.winfo_width(), root.winfo_height())
    x_coordinate = int((root.winfo_screenwidth() - root.winfo_width()) / 2)
    y_coordinate = int((root.winfo_screenheight() - root.winfo_height()) / 2)
    root.geometry(f'+{x_coordinate-7}+{y_coordinate-37}')

    root.mainloop()

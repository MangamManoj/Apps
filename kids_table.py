# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:32:06 2022

@author: manga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from num2words import num2words

st.title('Tables')
usr_ip = st.number_input('Insert the table no.',1)




def tables(x):
    #str= print( num2words(x).capitalize(),"Table","\n")
    st.write(num2words(x).capitalize(),"Table","\n")
    fig = plt.figure(figsize=(12,6))
    for i in range(0,10):
        st.write("%d x %d = %d" %(x,(i+1),x*(i+1)))
        plt.scatter((i+1),x*(i+1))
        plt.vlines((i+1),0, x*(i+1), linestyle="dashed")
        plt.hlines(x*(i+1),0,(i+1), linestyle="dashed",colors='red')
        
    plt.xticks(np.arange(1,11,1))
    plt.yticks(np.arange(x*1,x*(i+1)+1,x))
    plt.title('{0} {1}'.format(num2words(x).capitalize(), "Table"))
    st.pyplot(fig)
   
    return

tables(usr_ip)

    

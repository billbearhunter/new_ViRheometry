import numpy as np
import streamlit as st
class Param:
    def __init__(self, eta, n, sigmaY):
        self.eta    = eta
        self.n      = n
        self.sigmaY = sigmaY

    def vectorize(self):
        return np.array([self.eta, self.n, self.sigmaY])

    def display_status(self):
        print("*------------ Param ------------*")
        print("eta = ", self.eta)
        print("n = ", self.n)
        print("sigmaY  = ", self.sigmaY)

    def st_display_status(self):
        st.write("*------------ Param ------------*")
        st.write("eta = ", self.eta)
        st.write("n = ", self.n)
        st.write("sigmaY  = ", self.sigmaY)
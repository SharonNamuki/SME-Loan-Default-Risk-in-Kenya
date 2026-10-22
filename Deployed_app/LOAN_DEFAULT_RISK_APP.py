"""
LOAN DEFAULT RISK DETECTOR
3-Tier Risk Assessment System
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# ================== NUMPY _CORE FIX ==================
# Force numpy to have _core attribute
import numpy as np

# Check and fix numpy._core
if not hasattr(np, '_core'):
    print("FIXING: Creating numpy._core from numpy.core")
    
    # Import numpy.core
    import numpy.core
    
    # Create _core as an alias to core
    np._core = numpy.core
    
    # Also update sys.modules
    import sys
    sys.modules['numpy._core'] = numpy.core
    
    # Create submodules
    if hasattr(numpy.core, 'multiarray'):
        np._core.multiarray = numpy.core.multiarray
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
    
    if hasattr(numpy.core, 'umath'):
        np._core.umath = numpy.core.umath
        sys.modules['numpy._core.umath'] = numpy.core.umath
    
    if hasattr(numpy.core, '_multiarray_umath'):
        np._core._multiarray_umath = numpy.core._multiarray_umath
        sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath

print(f"After fix - Has numpy._core: {hasattr(np, '_core')}")
# =====================================================

# ================== PREPROCESSOR CLASS DEFINITION ==================
class LoanDeploymentPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for deployment
    Replicates exactly what was done during training
    """
    
    def __init__(self, deployment_package):
        self.deployment_package = deployment_package
        
        # Extract all transformers and config
        self.scaler = deployment_package.get('scaler')
        self.scaled_features = deployment_package.get('scaled_features', [])
        self.log_features = deployment_package.get('log_features', [])
        self.winsor_limits = deployment_package.get('winsor_limits', {})
        self.clip_limits = deployment_package.get('clip_limits', {})
        self.never_scale_features = deployment_package.get('never_scale_features', [])
        self.feature_names = deployment_package.get('feature_names', [])
        self.ohe = deployment_package.get('ohe')
        self.epsilon = deployment_package.get('epsilon', 0.001)
        
    def fit(self, X, y=None):
        # Already fitted during training
        return self
    
    def transform(self, X):
        """Apply all preprocessing steps"""
        df = X.copy()
        
        # 1. Fix zero values
        df = self._fix_zeros(df)
        
        # 2. Apply log transformations
        df = self._apply_log(df)
        
        # 3. Apply winsorization
        df = self._apply_winsorization(df)
        
        # 4. Apply clipping
        df = self._apply_clipping(df)
        
        # 5. Apply scaling
        df = self._apply_scaling(df)
        
        # 6. Ensure feature order
        df = self._ensure_feature_order(df)
        
        return df
    
    def _fix_zeros(self, df):
        """Replace zeros with epsilon"""
        for col in df.columns:
            if df[col].iloc[0] == 0:
                df[col] = self.epsilon
        return df
    
    def _apply_log(self, df):
        """Apply log1p to specified features"""
        for col in self.log_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return df
    
    def _apply_winsorization(self, df):
        """Clip extreme values"""
        for col, (low, high) in self.winsor_limits.items():
            if col in df.columns:
                if low is not None and high is not None:
                    df[col] = np.clip(df[col], low, high)
                elif high is not None:
                    df[col] = np.clip(df[col], df[col].min(), high)
                elif low is not None:
                    df[col] = np.clip(df[col], low, df[col].max())
        return df
    
    def _apply_clipping(self, df):
        """Apply clipping to age and account_age_days"""
        for col, (low, high) in self.clip_limits.items():
            if col in df.columns:
                if low is not None and high is not None:
                    df[col] = np.clip(df[col], low, high)
                elif high is not None:
                    df[col] = np.clip(df[col], df[col].min(), high)
                elif low is not None:
                    df[col] = np.clip(df[col], low, df[col].max())
        return df
    
    def _apply_scaling(self, df):
        """Apply RobustScaler to features that need it"""
        if self.scaler and self.scaled_features:
            # Ensure all scaled features exist in input
            existing_scaled = [col for col in self.scaled_features if col in df.columns]
            if existing_scaled:
                df[existing_scaled] = self.scaler.transform(df[existing_scaled])
        return df
    
    def _ensure_feature_order(self, df):
        """Add missing features with zeros and ensure correct order"""
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        # Return in correct order
        return df[self.feature_names]
    
    def transform_single(self, input_dict):
        """Transform a single sample from dictionary input"""
        # Convert dict to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Apply all transformations
        return self.transform(df)

# ================== STREAMLIT APP ==================

# Page configuration
st.set_page_config(
    page_title="Loan Risk Detector",
    page_icon="🏦",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 5px solid #059669;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .section-divider {
        margin: 30px 0;
        border-top: 2px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_deployment_package():
    """Load deployment package with model and preprocessor"""
    try:

        # Get current directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build paths
        model_path = os.path.join(current_dir, 'lightgbm_model.pkl')
        transformers_path = os.path.join(current_dir, 'deployment_transformers.pkl')
            
        if not os.path.exists(transformers_path):
            st.error(f"ERROR: Transformers file does not exist at {transformers_path}")
            return None, None, None, None
            
        
        # FIXED: Use the absolute paths, not relative paths!
        model = joblib.load(model_path)  
        deployment_transformers = joblib.load(transformers_path) 
        
        # Get feature names from deployment transformers
        feature_names = deployment_transformers['feature_names']
        
        # Create a fresh preprocessor instance
        preprocessor = LoanDeploymentPreprocessor(deployment_transformers)
        
        return model, preprocessor, feature_names, deployment_transformers
        
    except Exception as e:
        st.error(f"Error loading deployment package: {e}")
        return None, None, None, None

def create_input_features(
    age, account_age_days, loanamount, loan_term_months,
    num_ontime_payments, num_late_payments,
    num_penalties, previous_loan_count, previous_clearance_rate,
    loan_size_escalation, has_guarantor, gender, productid, loan_month
):
    """Create all 31 input features from UI inputs"""
    # Convert loan term months to days (matching training pipeline)
    loan_term_days = loan_term_months * 30
    
    total_payments = num_ontime_payments + num_late_payments
    
    if total_payments > 0:
        late_payment_rate = num_late_payments / total_payments
        penalty_rate = num_penalties / total_payments
    else:
        late_payment_rate = 0.5
        penalty_rate = 0
    
    has_payment_history = 1 if (num_late_payments > 0 or num_ontime_payments > 0) else 0
    avg_installment_paid = loanamount / loan_term_months if loan_term_months > 0 else loanamount / 12
    
    if productid == "Other":
        actual_product = "30111"
        product_infrequent = 1
    else:
        actual_product = productid
        product_infrequent = 0
    
    features = {
        'num_late_payments': float(num_late_payments),
        'num_ontime_payments': float(num_ontime_payments),
        'num_penalties': float(num_penalties),
        'late_payment_rate': float(late_payment_rate),
        'avg_installment_paid': float(avg_installment_paid),
        'penalty_rate': float(penalty_rate),
        'has_payment_history': float(has_payment_history),
        'age': float(age),
        'account_age_days': float(account_age_days),
        'previous_loan_count': float(previous_loan_count),
        'previous_clearance_rate': float(previous_clearance_rate),
        'loan_size_escalation': float(loan_size_escalation),
        'isloannew': 1.0,
        'has_guarantor': 1.0 if has_guarantor else 0.0,
        'gender': 1.0 if gender == 'Male' else 0.0,
        'productid_30111': 1.0 if actual_product == '30111' and product_infrequent == 0 else 0.0,
        'productid_30114': 1.0 if actual_product == '30114' and product_infrequent == 0 else 0.0,
        'productid_30150': 1.0 if actual_product == '30150' and product_infrequent == 0 else 0.0,
        'productid_30154': 1.0 if actual_product == '30154' and product_infrequent == 0 else 0.0,
        'productid_infrequent_sklearn': float(product_infrequent),
        'loan_month_2': 1.0 if loan_month == 2 else 0.0,
        'loan_month_3': 1.0 if loan_month == 3 else 0.0,
        'loan_month_4': 1.0 if loan_month == 4 else 0.0,
        'loan_month_5': 1.0 if loan_month == 5 else 0.0,
        'loan_month_6': 1.0 if loan_month == 6 else 0.0,
        'loan_month_7': 1.0 if loan_month == 7 else 0.0,
        'loan_month_8': 1.0 if loan_month == 8 else 0.0,
        'loan_month_9': 1.0 if loan_month == 9 else 0.0,
        'loan_month_10': 1.0 if loan_month == 10 else 0.0,
        'loan_month_11': 1.0 if loan_month == 11 else 0.0,
        'loan_month_12': 1.0 if loan_month == 12 else 0.0,
        # Add loan_term_days to match training features
        'loan_term_days': float(loan_term_days),
    }
    
    return features

def apply_complete_preprocessing(input_df, preprocessor):
    """Apply complete preprocessing using the preprocessor class"""
    return preprocessor.transform(input_df)

def get_risk_level(probability):
    """Convert probability to risk levels"""
    if probability < 40:
        return "LOW RISK", "#059669", "APPROVE", "🟢"
    elif probability < 70:
        return "MEDIUM RISK", "#D97706", "REVIEW", "🟡"
    else:
        return "HIGH RISK", "#DC2626", "TAKE ACTION", "🔴"

def main():
    
    # Header
    st.markdown('<div class="main-header">🏦 Loan Default Risk Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional 3-Tier Risk Assessment System</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    if 'test_case' not in st.session_state:
        st.session_state.test_case = {}
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("## 🎯 Quick Test Cases")
        st.markdown("---")
        
        test_cases = {
            "good": {
                "age": 35, "account_age_days": 800, "loanamount": 5000,
                "loan_term_months": 12, "num_ontime_payments": 24,
                "num_late_payments": 0, "num_penalties": 0,
                "previous_loan_count": 3, "previous_clearance_rate": 0.95,
                "loan_size_escalation": 1.1, "has_guarantor": "Yes",
                "gender": "Male", "productid": "30111", "loan_month": 3
            },
            "medium": {
                "age": 40, "account_age_days": 100, "loanamount": 50000,
                "loan_term_months": 10, "num_ontime_payments": 3,
                "num_late_payments": 3, "num_penalties": 0,
                "previous_loan_count": 0, "previous_clearance_rate": 0.65,
                "loan_size_escalation": 2.1, "has_guarantor": "No",
                "gender": "Female", "productid": "Other", "loan_month": 11
            },
            "bad": {
                "age": 22, "account_age_days": 90, "loanamount": 20000,
                "loan_term_months": 36, "num_ontime_payments": 3,
                "num_late_payments": 3, "num_penalties": 4,
                "previous_loan_count": 0, "previous_clearance_rate": 0.0,
                "loan_size_escalation": 4.0, "has_guarantor": "No",
                "gender": "Male", "productid": "Other", "loan_month": 1
            }
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Good", use_container_width=True):
                st.session_state.test_case = test_cases["good"]
                st.rerun()
        with col2:
            if st.button("Medium", use_container_width=True):
                st.session_state.test_case = test_cases["medium"]
                st.rerun()
        with col3:
            if st.button("Bad", use_container_width=True):
                st.session_state.test_case = test_cases["bad"]
                st.rerun()
        
        st.markdown("---")
        
        # Debug toggle
        debug = st.checkbox("Enable Debug Mode", value=False)
        
        # Load model info
        model, preprocessor, feature_names, deployment_transformers = load_deployment_package()
        if model and preprocessor:
            st.markdown("### 📊 Model Information")
            st.info(f"""
            **Features:** {len(feature_names)}  
            **Preprocessing:** Complete pipeline loaded
            **Risk Thresholds:**
            - 🟢 LOW: 0-40% default
            - 🟡 MEDIUM: 40-70% default  
            - 🔴 HIGH: 70-100% default
            """)
            
            if debug:
                with st.expander("Debug: View Feature Names"):
                    st.write(f"First 10 features: {feature_names[:10]}")
                    st.write(f"Total features: {len(feature_names)}")
    
    # Input form sections
    st.markdown("## 👤 Borrower Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, 
                             value=st.session_state.get('test_case', {}).get('age', 35))
        account_age_days = st.number_input("Account Age (days)", min_value=0, max_value=3650,
                                          value=st.session_state.get('test_case', {}).get('account_age_days', 800))
    
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female"], 
                             index=0 if st.session_state.get('test_case', {}).get('gender', "Male") == "Male" else 1)
        has_guarantor = st.selectbox("Has Guarantor", options=["Yes", "No"],
                                    index=0 if st.session_state.get('test_case', {}).get('has_guarantor', "Yes") == "Yes" else 1)
    
    with col3:
        previous_loan_count = st.number_input("Previous Loans", min_value=0, max_value=50,
                                             value=st.session_state.get('test_case', {}).get('previous_loan_count', 3))
        previous_clearance_rate = st.slider("Previous Clearance Rate", min_value=0.0, max_value=1.0,
                                           value=st.session_state.get('test_case', {}).get('previous_clearance_rate', 0.85), step=0.01)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 💰 Loan Information")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        loanamount = st.number_input("Loan Amount (KSH)", min_value=100, max_value=100000,
                                    value=st.session_state.get('test_case', {}).get('loanamount', 5000))
        loan_term_months = st.number_input("Loan Term (months)", min_value=1, max_value=60,
                                          value=st.session_state.get('test_case', {}).get('loan_term_months', 12))
    
    with col5:
        loan_size_escalation = st.slider("Loan Size Escalation", min_value=0.1, max_value=5.0,
                                        value=st.session_state.get('test_case', {}).get('loan_size_escalation', 1.2), step=0.1)
        productid = st.selectbox("Product ID", 
                                options=["30111", "30114", "30150", "30154", "Other"],
                                index=["30111", "30114", "30150", "30154", "Other"]
                                .index(st.session_state.get('test_case', {}).get('productid', "30111")))
    
    with col6:
        loan_month = st.selectbox("Loan Month", options=list(range(1, 13)),
                                 index=st.session_state.get('test_case', {}).get('loan_month', 3) - 1)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 📊 Payment History")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        num_ontime_payments = st.number_input("On-time Payments", min_value=0, max_value=100,
                                             value=st.session_state.get('test_case', {}).get('num_ontime_payments', 10))
    
    with col8:
        num_late_payments = st.number_input("Late Payments", min_value=0, max_value=100,
                                           value=st.session_state.get('test_case', {}).get('num_late_payments', 2))
    
    with col9:
        num_penalties = st.number_input("Penalties", min_value=0, max_value=50,
                                       value=st.session_state.get('test_case', {}).get('num_penalties', 1))
    
    # Quick metrics
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📈 Quick Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        total_payments = num_ontime_payments + num_late_payments
        late_rate = (num_late_payments / total_payments * 100) if total_payments > 0 else 0
        st.metric("Late Payment Rate", f"{late_rate:.1f}%")
    
    with metric_col2:
        avg_installment = loanamount / loan_term_months if loan_term_months > 0 else loanamount / 12
        st.metric("Monthly Installment", f"KES {avg_installment:,.0f}")
    
    with metric_col3:
        clearance_color = "🟢" if previous_clearance_rate >= 0.8 else "🟡" if previous_clearance_rate >= 0.6 else "🔴"
        st.metric("Clearance Rate", f"{clearance_color} {previous_clearance_rate:.0%}")
    
    with metric_col4:
        escalation_status = "🟢" if loan_size_escalation < 1.5 else "🟡" if loan_size_escalation < 2.5 else "🔴"
        st.metric("Escalation", f"{escalation_status} {loan_size_escalation:.1f}x")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Main Predict Button
    if st.button("🚀 Assess Default Risk", type="primary", use_container_width=True):
        st.session_state.analyze_clicked = True
    
    # Display results after button click
    if hasattr(st.session_state, 'analyze_clicked') and st.session_state.analyze_clicked:
        if model is None or preprocessor is None:
            st.error("Model or preprocessor failed to load. Please check the configuration.")
            return
        
        with st.spinner("Analyzing risk profile..."):
            # Create features
            input_features = create_input_features(
                age=age,
                account_age_days=account_age_days,
                loanamount=loanamount,
                loan_term_months=loan_term_months,
                num_ontime_payments=num_ontime_payments,
                num_late_payments=num_late_payments,
                num_penalties=num_penalties,
                previous_loan_count=previous_loan_count,
                previous_clearance_rate=previous_clearance_rate,
                loan_size_escalation=loan_size_escalation,
                has_guarantor=(has_guarantor == "Yes"),
                gender=gender,
                productid=productid,
                loan_month=loan_month
            )
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_features])
            
            if debug:
                with st.expander("Debug: Raw Input Features"):
                    st.write("Raw features before preprocessing:")
                    st.dataframe(input_df)
            
            # Apply COMPLETE preprocessing using the preprocessor
            processed_input = apply_complete_preprocessing(input_df, preprocessor)
            
            if debug:
                with st.expander("Debug: Processed Features"):
                    st.write("Features after complete preprocessing:")
                    st.dataframe(processed_input)
                    
                    # Show which transformations were applied
                    st.write("Preprocessing applied:")
                    st.write(f"- Log features: {preprocessor.log_features}")
                    st.write(f"- Winsor limits: {preprocessor.winsor_limits}")
                    st.write(f"- Clip limits: {preprocessor.clip_limits}")
                    st.write(f"- Scaled features: {preprocessor.scaled_features}")
            
            # Make prediction
            probabilities = model.predict_proba(processed_input)[0]
            prob_default = probabilities[1] * 100
            
            # Get risk level
            risk_level, risk_color, recommendation, emoji = get_risk_level(prob_default)
            
            # Store results
            st.session_state.prob_default = prob_default
            st.session_state.risk_level = risk_level
            st.session_state.risk_color = risk_color
            st.session_state.recommendation = recommendation
            st.session_state.emoji = emoji
    
    # Display results if available
    if hasattr(st.session_state, 'prob_default'):
        st.markdown("---")
        st.markdown("## 📋 Risk Assessment Results")
        
        # Risk score display
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"### {st.session_state.emoji}")
            st.markdown(f"# {st.session_state.risk_level}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_r2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown("### Default Probability")
            st.markdown(f"# {st.session_state.prob_default:.1f}%")
            st.markdown(f"**Recommendation:** {st.session_state.recommendation}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.prob_default,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Risk Score", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': st.session_state.risk_color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': st.session_state.prob_default
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors analysis
        st.markdown("### ⚠️ Risk Factor Analysis")
        
        risk_class = f"risk-{st.session_state.risk_level.split()[0].lower()}"
        st.markdown(f'<div class="risk-card {risk_class}">', unsafe_allow_html=True)
        
        risk_factors = []
        
        # Calculate metrics
        total_payments = num_ontime_payments + num_late_payments
        late_rate = num_late_payments / max(total_payments, 1)
        
        # Payment behavior
        if late_rate > 0.25:
            risk_factors.append("❌ High late payment rate (>25%)")
        elif late_rate > 0.1:
            risk_factors.append("⚠️ Moderate late payments (10-25%)")
        else:
            risk_factors.append("✅ Good payment history (<10% late)")
        
        # Clearance rate
        if previous_clearance_rate < 0.7:
            risk_factors.append("❌ Low clearance rate (<70%)")
        elif previous_clearance_rate < 0.85:
            risk_factors.append("⚠️ Moderate clearance rate (70-85%)")
        else:
            risk_factors.append("✅ High clearance rate (>85%)")
        
        # Loan escalation
        if loan_size_escalation > 2.0:
            risk_factors.append("❌ High loan escalation (>2.0x)")
        elif loan_size_escalation > 1.5:
            risk_factors.append("⚠️ Moderate escalation (1.5-2.0x)")
        else:
            risk_factors.append("✅ Conservative escalation (<1.5x)")
        
        # Account age
        if account_age_days < 180:
            risk_factors.append("⚠️ New account (<6 months)")
        elif account_age_days < 365:
            risk_factors.append("✅ Established account (6-12 months)")
        else:
            risk_factors.append("✅ Long-term account (>1 year)")
        
        # Guarantor
        if has_guarantor == "No":
            risk_factors.append("⚠️ No guarantor")
        else:
            risk_factors.append("✅ Has guarantor")
        
        # Penalties
        if num_penalties > 0:
            risk_factors.append("⚠️ Has penalties")
        else:
            risk_factors.append("✅ No penalties")
        
        # Display risk factors
        for factor in risk_factors:
            if "❌" in factor:
                st.error(factor)
            elif "⚠️" in factor:
                st.warning(factor)
            else:
                st.success(factor)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action recommendations
        st.markdown("### 🎯 Recommendation Summary")
        
        if st.session_state.risk_level == "LOW RISK":
            st.success("""
            **APPROVE** - Borrower demonstrates strong creditworthiness.
            
            **Suggested Actions:**
            - Process application with standard terms
            - Regular monitoring schedule
            - Consider for credit limit increases
            """)
        elif st.session_state.risk_level == "MEDIUM RISK":
            st.warning("""
            **MONITOR** - Moderate risk profile requires careful review.
            
            **Suggested Actions:**
            - Additional financial review
            - Consider higher interest rate (1-2% above standard)
            - Monthly payment reviews for first 6 months
            - Consider requiring guarantor
            """)
        else:
            st.error("""
            **TAKE ACTION** - High risk profile identified.
            
            **Suggested Actions:**
            - Detailed financial review required
            - Consider much smaller loan amount
            - Strong collateral or guarantor required
            - Consider credit counseling referral
            """)
        
        # Debug information
        if debug and deployment_transformers:
            st.markdown("---")
            st.markdown("### 🔍 Technical Details")
            
            with st.expander("View Model Information"):
                st.write(f"**Model Output:**")
                st.write(f"- Probability of no default: {100 - st.session_state.prob_default:.2f}%")
                st.write(f"- Probability of default: {st.session_state.prob_default:.2f}%")
                
                st.write(f"**Key Input Features:**")
                st.write(f"- Age: {age}")
                st.write(f"- Account Age: {account_age_days} days")
                st.write(f"- Previous Loans: {previous_loan_count}")
                st.write(f"- Clearance Rate: {previous_clearance_rate:.1%}")
                st.write(f"- Late Payment Rate: {late_rate:.1%}")
                st.write(f"- Loan Escalation: {loan_size_escalation:.1f}x")
                st.write(f"- Has Guarantor: {has_guarantor}")
                st.write(f"- Product Type: {productid}")
                
                st.write(f"**Model Information:**")
                st.write(f"- Total Features: {len(feature_names)}")
                st.write(f"- Risk Thresholds: 0-40% (LOW), 40-70% (MEDIUM), 70-100% (HIGH)")
                
                st.write(f"**Preprocessing Configuration:**")
                st.write(f"- Log transformed features: {len(preprocessor.log_features)}")
                st.write(f"- Winsorized features: {len(preprocessor.winsor_limits)}")
                st.write(f"- Scaled features: {len(preprocessor.scaled_features) if preprocessor.scaled_features else 0}")
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About This System")
    st.write("""
    This loan risk assessment system uses machine learning to predict default probability based on carefully selected features including:
    - Payment history and behavior
    - Borrower profile and demographics  
    - Loan characteristics and terms
    - Historical performance metrics
    
    **All predictions are based on:**
    - Complete preprocessing pipeline (log transforms, winsorization, clipping, scaling)
    - LightGBM machine learning model
    - 3-tier risk assessment (Low/Medium/High)
    - Historical data patterns and statistical modeling
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

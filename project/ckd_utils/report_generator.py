from fpdf import FPDF
import datetime

def generate_detailed_recommendations(input_dict, stage_label):
    """
    Generate detailed health recommendations based on patient input and CKD stage.
    """
    recommendations = []
    
    # 1. Base stage recommendation
    recs = {
        "Healthy Kidney":
            "Great news! Your kidneys appear healthy. Maintain a balanced diet, stay hydrated, "
            "exercise regularly, and get annual check-ups.",
        "Mild CKD (Stage 1–2)":
            "Early-stage CKD detected. Your kidneys are still functioning well. Consult a nephrologist, "
            "control blood pressure & blood sugar, and limit sodium intake.",
        "Moderate CKD (Stage 3)":
            "Moderate CKD Stage 3 detected. Kidney function is moderately reduced. Immediate medical "
            "attention is advised. Follow a kidney-friendly diet and monitor labs regularly.",
        "Severe CKD (Stage 4)":
            "Severe CKD Stage 4 detected. Kidney function is severely reduced. Begin planning for renal "
            "replacement therapy (dialysis or transplant). Work closely with your care team.",
        "Kidney Failure (Stage 5)":
            "Kidney Failure (Stage 5) detected. Immediate medical intervention is required. "
            "Dialysis or kidney transplant may be necessary. Contact your nephrologist immediately.",
    }
    recommendations.append(("Overall Assessment", recs.get(stage_label, "Please consult a medical professional.")))
    
    # 2. Blood Pressure
    sys_bp = input_dict.get("Systolic_BP", 120)
    dia_bp = input_dict.get("Diastolic_BP", 80)
    if sys_bp >= 130 or dia_bp >= 80:
        recommendations.append(("Blood Pressure Control", 
            "Your blood pressure is elevated. High blood pressure can accelerate kidney damage. "
            "Aim for < 130/80 mmHg. Reduce sodium (salt) intake, engage in regular aerobic exercise, and consult your doctor for potential medication adjustments."))
    
    # 3. BMI
    bmi = input_dict.get("BMI", 25)
    if bmi >= 30:
        recommendations.append(("Weight Management",
            "Your BMI indicates obesity, which increases the workload on your kidneys. "
            "Consider a structured weight loss plan including a healthy diet and regular physical activity to lower your risk."))
    elif bmi >= 25:
        recommendations.append(("Weight Management",
            "Your BMI indicates you are overweight. Gradually reducing weight can significantly improve your kidney and cardiovascular health."))
            
    # 4. Glucose / Diabetes
    diabetes = input_dict.get("Diabetes", "No")
    hba1c = input_dict.get("HbA1c", 5.5)
    if diabetes == "Yes" or hba1c >= 6.5:
        recommendations.append(("Blood Sugar Management",
            "Poor blood sugar control is a leading cause of kidney disease progression. "
            "Strictly manage your blood glucose levels, monitor your HbA1c, and adhere to any prescribed diabetic treatments."))
            
    # 5. Diet specifics (Sodium/Potassium based on CKD Stage)
    if "Stage 3" in stage_label or "Stage 4" in stage_label or "Stage 5" in stage_label:
        recommendations.append(("Dietary Restrictions",
            "Due to advanced CKD, strictly limit dietary sodium, phosphorus, and potassium. Avoid processed foods, limit dairy and certain fruits/vegetables as directed by a renal dietitian."))
            
    # 6. Smoking
    smoking = input_dict.get("Smoking_Status", "No")
    if smoking == "Yes":
        recommendations.append(("Smoking Cessation",
            "Smoking severely damages blood vessels and worsens kidney disease. Quitting smoking is one of the most effective steps you can take to protect your kidneys."))
            
    return recommendations


class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(43, 108, 176) # Theme color
        self.cell(0, 10, "Chronic Kidney Disease Assessment Report", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def create_pdf_report(input_dict, label, conf, recommendations):
    """
    Create a PDF report using fpdf2 and return the binary string to be downloaded.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Date
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 10, f"Date: {current_date}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 1. Patient Demographics & Vitals
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(26, 32, 44)
    pdf.cell(0, 10, "1. Patient Profile", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    
    gender_str = "Male" if input_dict.get("Gender") == 1 else "Female"
    
    details = [
        f"Age: {input_dict.get('Age')} years",
        f"Gender: {gender_str}",
        f"BMI: {input_dict.get('BMI')}",
        f"Blood Pressure: {input_dict.get('Systolic_BP')}/{input_dict.get('Diastolic_BP')} mmHg",
        f"Diabetes: {input_dict.get('Diabetes')}",
        f"Hypertension: {input_dict.get('Hypertension')}",
        f"Smoking Status: {input_dict.get('Smoking_Status')}",
    ]
    
    # Format into two columns if possible or just a simple list
    for item in details:
        pdf.cell(0, 6, f"- {item}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    
    # 2. Prediction Result
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. CKD Prediction Result", new_x="LMARGIN", new_y="NEXT")
    
    # Highlight the result box
    pdf.set_fill_color(235, 244, 255) # Light blue background
    pdf.set_font("helvetica", "B", 12)
    
    safe_label = str(label).replace('–', '-')
    res_text = f"Predicted Stage: {safe_label}"
    if conf is not None:
        res_text += f"   (Confidence: {conf:.1f}%)"
        
    pdf.cell(0, 12, res_text, border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # 3. Personalized Health Recommendations
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Personalized Health Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    for title, desc in recommendations:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(43, 108, 176)
        safe_title = str(title).replace('–', '-')
        pdf.cell(0, 8, safe_title, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(50)
        safe_desc = str(desc).replace('–', '-')
        pdf.multi_cell(0, 6, safe_desc, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        
    # 4. Disclaimer
    pdf.ln(10)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(150, 0, 0)
    pdf.multi_cell(0, 6, "Medical Disclaimer: This report is generated by an AI prediction system for educational and informational purposes only. It is NOT a medical diagnosis and should not replace professional medical advice. Always consult a qualified healthcare provider regarding your health conditions.", new_x="LMARGIN", new_y="NEXT")
    
    return pdf.output()
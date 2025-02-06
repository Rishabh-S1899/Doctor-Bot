import ollama
import json
file_name = "../PMC-Patients.json"
with open(file_name,"r") as f:
    data = json.load(f)
patient_data = []
for patient in data:
    patient_data.append(patient["patient"])
# print(patient_data[0])



patient_data_disease = []
for patient_description in patient_data:
    if "cough" in patient_description or "cold" in patient_description or "fever" in patient_description:
        patient_data_disease.append(patient_description)

with open("data_disease.json","a") as file:
    responses = []
    question = "If the patient is human and has cough cold or fever, list the symtopms for it from the context only and nothing else along with the disease they had and it's label, if not present or if not human then no output"
    for i in range(0,250):
        context = patient_data_disease[i]
        response = ollama.chat(model="llama3.2", messages=[{
            "role":"user",
            "content":f"{context}, {question}"
        }])
        responses.append({
            "response": response["message"]["content"]
        })
        if (i+1)%50 == 0:
            try:
                json.dump(responses,file,indent=4)
                responses.clear()
                print(i,"done")
            except: 
                break
        if (i+1)%500 == 0:
            break




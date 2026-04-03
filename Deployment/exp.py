DATE_COLUMNS = {
    'admission': ('AdmissionDt', 'DischargeDt', 'IP_Number_of_Days_in_Hospital'),
    'claim':     ('ClaimStartDt', 'ClaimEndDt', 'IP_Claim_Days'),
}

for start_col, end_col in DATE_COLUMNS.values():
    print(start_col,end_col)
import tkinter 
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
def save_frame():
    print("Saving the information")
    firstname = first_name_entry.get()
    lastname = last_name_entry.get()
    Occupation = title_combobox.get()
    age = age_spinbox.get()
    organisms = pets_combobox.get()
    ACstatus = AC_status_var.get()
    print(f"name: {firstname} {lastname} worker as: {Occupation} age: {age}, organisms at house: {organisms}")
    print("AC status: ", ACstatus)

tempIn = 100 # call variable from the other file and display it here

def increaseTemp():
    # increase temperature by 1 degree
    tempIn = tempIn.get() 
    tempIn = tempIn + 1

    
window = tkinter.Tk() # parent window - largest box
window.title("ThermoFli Dashboard")


# create a frame
frame = tkinter.Frame(window)
frame.pack()

#user info frame
user_info_frame = tkinter.LabelFrame(frame, text="User Info")
user_info_frame.grid(row=0, column=0, padx=20, pady=10)

# taking user information

# First/last name creating input box and label
first_name_label = tkinter.Label(user_info_frame, text="First Name:")
first_name_label.grid(row=0, column=0)
last_name_label = tkinter.Label(user_info_frame, text="Last Name:")
last_name_label.grid(row=0, column=1)

first_name_entry = tkinter.Entry(user_info_frame)
first_name_entry.grid(row=1, column=0)
last_name_entry = tkinter.Entry(user_info_frame)
last_name_entry.grid(row=1, column=1)

# occupation type input box and label
title_lable= tkinter.Label(user_info_frame, text="Occupation Type:")
title_combobox = ttk.Combobox(user_info_frame, values=["Student", "Worker", "Retired", "Hybrid Worker"])
title_lable.grid(row=0, column=2)
title_combobox.grid(row=1, column=2)

# age input box and label
age_label = tkinter.Label(user_info_frame, text="Age:")
age_spinbox = tkinter.Spinbox(user_info_frame, from_= 1, to=110)
age_label.grid(row=0, column=3)
age_spinbox.grid(row=1, column=3)

# pets input box and label
Pets_label = tkinter.Label(user_info_frame, text="Other living organisms at home:")
pets_combobox = ttk.Combobox(user_info_frame, values=["--NA--","Plants", "Dogs", "Cats", "Baby at home"])
Pets_label.grid(row=2 ,column=0)
pets_combobox.grid(row=3, column=0) 

#setting the padding for the widgets
for widget in user_info_frame.winfo_children():
    widget.grid_configure(padx=15, pady=10)
    
#saving the information
main_settings = tkinter.LabelFrame(frame, text="Temperature Settings")
main_settings.grid(row=1, column=0, sticky="news", padx=20, pady=20)

AC_status_var = tkinter.StringVar(value="OFF")
AC_status_label = tkinter.Label(main_settings, text=" AC Status:", )
AC_status_check = tkinter.Checkbutton(main_settings,variable = AC_status_var, onvalue="ON", offvalue="OFF")
AC_status_label.grid(row=0, column=0)
AC_status_check.grid(row=0, column=1)

for widget in user_info_frame.winfo_children():
    widget.grid_configure(padx=15, pady=10)
    
model_frame = tkinter.LabelFrame(frame, text=" Your model")
model_frame.grid(row=2, column=0, sticky="news", padx=20, pady=20)

# creating ubcrease temperature button
# creating decrease temperature button
first_name_label = tkinter.Label(model_frame, text=" Change Temperature: ")
first_name_label.grid(row=0, column=0,padx=15, pady=10)

increase_temp_button = tkinter.Button(model_frame, text ="                 +               ", command=increaseTemp)
increase_temp_button.grid(row=1, column=1,padx=15, pady=10)


AC_display = tkinter.Label(model_frame, text= tempIn)
AC_display.grid(row=1, column=2,padx=15, pady=10)

decrease_temp_button = tkinter.Button(model_frame, text ="                 -               ")
decrease_temp_button.grid(row=1, column=3,padx=15, pady=10)



# save changes button
save_frame_button = tkinter.Button(frame, text ="Save", command=save_frame) 
save_frame_button.grid(row=3, column=0,sticky="news", padx=15, pady=10)

#displaying the graph

def graph():
    # basically call matplotlib.pyplot from the otehr file and display it here
    houseprices = np.random.normal(6500, 1000, 5000)
    plt.hist(houseprices, 50)
    plt.show()


mybuttonn = tkinter.Button(window, text="Graph", command=graph, width=10, height=2)
mybuttonn.pack()

window.mainloop() 




    










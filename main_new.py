import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pytesseract
from datetime import datetime
from PIL import Image, ImageTk
import easyocr
import os
import re
import pandas as pd
from tkinter import messagebox
import sys
import sqlite3
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QComboBox, QStackedWidget, \
    QPushButton, QLineEdit, QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt
import logging
from logging.handlers import TimedRotatingFileHandler
import time
from datetime import timedelta


latest_camera_value = None

conn = sqlite3.connect('camera_settings.db')
cursor = conn.cursor()
cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_settings (
            id INTEGER PRIMARY KEY,
            camera_type TEXT,
            camera_value INT,
            gap_second INT
        )
    ''')
cursor.execute('''
    INSERT OR IGNORE INTO camera_settings (id, camera_type, camera_value, gap_second)
    VALUES (1, 'default', 0, 60)
''')
conn.commit()
conn.close()



# Configure logging to write to a file
logging.basicConfig(filename='my_log_file.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Replace print statements with logging statements
def print(*args, **kwargs):
    logging.info(' '.join(map(str, args)))


# Example usage of print replacement
print("This will be written to the log file")

# You can also use other logging levels
logging.warning("This is a warning message")
logging.error("This is an error message")

reader = easyocr.Reader(['en'])


def is_indian_number_plate(plate_number):
    # Regular expression to match Indian number plate formats
    # Modify this pattern based on the specific number plate format in your region
    pattern = (r'^(AP|AR|AS|BR|CG|GA|GJ|HR|HP|JH|KA|KL|MP|MH|MN|ML|MZ|NL|OD|PB|RJ|SK|TN|TS|TR|UP|UK|WB|AN|CH|DD|DL|LD'
               r'|LDN|PY)\s(\d{1,2})\s([A-Z]{1,2})\s(\d{1,4})$')
    return re.match(pattern, plate_number.upper()) is not None


def preprocess_number_plate(plate_number):
    # Remove spaces, hyphens, and other special characters from the number plate
    plate_number = re.sub(r'[^A-Z0-9]', '', plate_number.upper())

    print(plate_number)

    # Add spaces between the character groups to match the format in `is_indian_number_plate` function
    if len(plate_number) == 10:
        plate_number = plate_number[:2] + ' ' + plate_number[2:4] + ' ' + plate_number[4:6] + ' ' + plate_number[6:]
    return plate_number


# Set to store unique valid number plates
unique_valid_number_plates = set()

# Create a folder to save the scanned number plate images
output_folder = "scanned_number_plates"
os.makedirs(output_folder, exist_ok=True)


def save_scanned_number_plate_image(image, plate_number):
    filename = f"{output_folder}/{plate_number}.png"
    cv2.imwrite(filename, image)
    print(f"Scanned number plate saved: {filename}")


def process_frame():
    global display_new_plates
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to preprocess the frame
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Load the Haar Cascade for number plate detection
        plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

        # Detect the number plates using the Haar Cascade
        plates = plate_cascade.detectMultiScale(thresh, scaleFactor=1.1, minNeighbors=5, minSize=(100, 25))

        # Initialize a list to collect valid number plates
        valid_number_plates = []

        # Loop over the detected plates
        for (x, y, w, h) in plates:
            # Crop the region of interest (ROI) and apply OCR to it
            plate_roi = thresh[y:y + h, x:x + w]

            # OCR using EasyOCR
            results = reader.readtext(plate_roi, detail=0)

            # Process all possible recognized texts
            for text in results:
                # Remove extra characters and auto-correct
                text = ''.join(e for e in text if e.isalnum())
                text = text.upper()

                # Preprocess the recognized number plate
                converted_plate = preprocess_number_plate(text)

                # Check if the processed plate is in a valid Indian number plate format
                is_valid_format = is_indian_number_plate(converted_plate)

                if is_valid_format:
                    valid_number_plates.append(converted_plate)

                    # Draw a rectangle around the plate
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save the scanned number plate image
                    save_scanned_number_plate_image(frame[y:y + h, x:x + w], converted_plate)

                    # Insert data into the database with the specified gap_seconds
                    insert_data_into_db(converted_plate, gap_seconds)

                    # Update the table display with new data
                    populate_table()

                    # Update the table display with new data if allowed
                    if display_new_plates:
                        populate_table()

    # Display the processed frame with the rectangle
    processed_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(processed_frame_rgb)
    img = img.resize((640, 480), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    video_label.config(image=img)
    video_label.image = img


# Create a dictionary to store the last time a number plate was added to the database
last_added_times = {}

conn = sqlite3.connect('camera_settings.db')
cursor = conn.cursor()

# Retrieve the gap_second value from the table
cursor.execute('SELECT gap_second FROM camera_settings WHERE id = 1')
row = cursor.fetchone()

gap_seconds = row[0]
print(f"Retrieved gap_second value: {gap_seconds}")


# Set the time gap in seconds (e.g., 10 seconds)
# gap_seconds = 60
# print("gap_seconds")


# Initialize a SQLite database and cursor
connection = sqlite3.connect("number_plates.db")
cursor = connection.cursor()

# Check if the database file already exists
db_file = "number_plates.db"

if not os.path.exists(db_file):
    # Create a new database if it doesn't exist
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

# Create the number_plates table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS number_plates (
        id INTEGER PRIMARY KEY,
        number_plate TEXT,
        time TIMESTAMP,
        owner_name TEXT,
        number TEXT
    )
''')
connection.commit()


# Get the absolute path to the database file
# db_file_path = os.path.abspath("number_plate.db")
# connection = sqlite3.connect("D:\video plate\number_plates.db")

last_added_times = {}




def insert_data_into_db(number_plate, gap_seconds):
    # Get the current date and time(camera_var, add_gap_value, fps_val)
    current_time = datetime.now()
    time_str = current_time.strftime("%d-%m-%Y %H:%M:%S")

    # Get Owner Name and Number from the dictionary, or use default values if not found
    owner_name, number = number_plate_mapping.get(number_plate, ('Not Match', 'Not Match'))[:2]

    # Check if the number plate was added within the time gap
    if number_plate in last_added_times and (current_time - last_added_times[number_plate]) < timedelta(
            seconds=gap_seconds):
        print("Number plate already added within the time gap.")
        return

    # Insert data into the database
    cursor.execute("INSERT INTO number_plates (number_plate, time, owner_name, number) VALUES (?, ?, ?, ?)",
                   (number_plate, time_str, owner_name, number))
    connection.commit()

    # Update the last added time for the number plate
    last_added_times[number_plate] = current_time


# Close the database connection when done
connection.close()



# Function to update the video stream
def update_video_stream():
    ret, frame = cap.read()
    if ret:
        # Process the frame (you'll implement this function)
        process_frame()
        # Update the video_label with the processed frame
        # Use tkinter PhotoImage to display the frame in a Label
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_tk = ImageTk.PhotoImage(frame)
        video_label.img = frame_tk
        video_label.config(image=frame_tk)
        video_label.after(10, update_video_stream)  # Update every 10 milliseconds
    else:
        # Handle the case when the video stream ends
        cap.release()


display_new_plates = True  # Flag to control displaying new plates

number_plate_filter_entry = None
time_filter_entry = None
owner_name_filter_entry = None


def write_unique_plates_to_file():
    with open('valid_number_plates.txt', 'w') as file:
        for plate in unique_valid_number_plates:
            file.write(plate + '\n')







def open_member_management():
    member_management_window = tk.Toplevel(root)
    member_management_window.title("Member Management")
    member_management_window.geometry("600x400")

    # Create a frame for input fields
    input_frame = tk.Frame(member_management_window)
    input_frame.pack()

    # Create input fields for Number Plate, Owner Name, and Number
    # Create input fields for Flate Number and Vehicle
    flate_number_frame = tk.Frame(input_frame)
    flate_number_label = tk.Label(flate_number_frame, text="Flate Number")
    flate_number_entry = tk.Entry(flate_number_frame)
    flate_number_label.pack(side=tk.LEFT, padx=10, pady=5)
    flate_number_entry.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
    flate_number_frame.pack(anchor="w")

    vehicle_frame = tk.Frame(input_frame)
    vehicle_label = tk.Label(vehicle_frame, text="Vehicle           ")
    vehicle_entry = tk.Entry(vehicle_frame)
    vehicle_label.pack(side=tk.LEFT, padx=10, pady=5)
    vehicle_entry.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
    vehicle_frame.pack(anchor="w")

    number_plate_frame = tk.Frame(input_frame)
    number_plate_label = tk.Label(number_plate_frame, text="Number Plate")
    number_plate_entry = tk.Entry(number_plate_frame)
    number_plate_label.pack(side=tk.LEFT, padx=10, pady=5)
    number_plate_entry.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
    number_plate_frame.pack(anchor="w")

    owner_name_frame = tk.Frame(input_frame)
    owner_name_label = tk.Label(owner_name_frame, text="Owner Name ")
    owner_name_entry = tk.Entry(owner_name_frame)
    owner_name_label.pack(side=tk.LEFT, padx=10, pady=5)
    owner_name_entry.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
    owner_name_frame.pack(anchor="w")

    number_frame = tk.Frame(input_frame)
    number_label = tk.Label(number_frame, text="Number         ")
    number_entry = tk.Entry(number_frame)
    number_label.pack(side=tk.LEFT, padx=10, pady=5)
    number_entry.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
    number_frame.pack(anchor="w")

    # Load the existing data from the Excel file
    data_frame = pd.read_excel('faltedata.xlsx')

    def browse_file():
        global data_frame, number_plate_mapping

        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            new_data_frame = pd.read_excel(file_path)

            # Create a dictionary to map Number Plate to Owner Name and Number from new data
            new_number_plate_mapping = dict(zip(new_data_frame['Number Plate'], new_data_frame[
                ['Owner Name', 'Number', 'Flate Number', 'Vehicle']].values))

            # Update existing data with new or updated values
            for number_plate, details in new_number_plate_mapping.items():
                if number_plate in number_plate_mapping:
                    # Update existing member's data
                    data_frame.loc[data_frame['Number Plate'] == number_plate, 'Owner Name'] = details[0]
                    data_frame.loc[data_frame['Number Plate'] == number_plate, 'Number'] = details[1]
                else:
                    # Add new member's data to the DataFrame
                    data_frame.loc[len(data_frame)] = [details[2], details[3], number_plate, details[0], details[1]]

            # Update the global number_plate_mapping
            number_plate_mapping = dict(
                zip(data_frame['Number Plate'], data_frame[['Owner Name', 'Number', 'Flate Number', 'Vehicle']].values))

            # Save the updated DataFrame to the Excel file
            data_frame.to_excel('faltedata.xlsx', index=False)

            # Update the table display
            populate_table()

    def edit_member():
        number_plate = number_plate_entry.get()
        new_owner_name = owner_name_entry.get()
        new_number = number_entry.get()

        # Update member details in the DataFrame
        data_frame.loc[data_frame['Number Plate'] == number_plate, 'Owner Name'] = new_owner_name
        data_frame.loc[data_frame['Number Plate'] == number_plate, 'Number'] = new_number

        # Save the DataFrame to the Excel file
        data_frame.to_excel('faltedata.xlsx', index=False)

        # Clear the input fields
        number_plate_entry.delete(0, tk.END)
        owner_name_entry.delete(0, tk.END)
        number_entry.delete(0, tk.END)

        # Show success alert
        messagebox.showinfo("Member Edited", "Member details edited successfully.")

    def download_excel_file():
        # Read the data from the Excel file
        d_file_name = "Demo Excel.xlsx"
        data_frame = pd.read_excel('Demo Excel.xlsx')

        # Display a file save dialog
        download_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")],
                                                     initialfile=d_file_name)

        # If the user didn't cancel the dialog
        if download_path:
            # Save the data frame to the selected path
            data_frame.to_excel(download_path, index=False)
            print(f"Excel file downloaded to: {download_path}")

    def delete_member():
        number_plate = number_plate_entry.get()

        if number_plate:  # Check if the input is not empty
            data_frame.drop(data_frame[data_frame['Number Plate'] == number_plate].index, inplace=True)
            data_frame.to_excel('faltedata.xlsx', index=False)

            # Clear the input field after deletion
            number_plate_entry.delete(0, tk.END)

            # Show success alert
            messagebox.showinfo("Member Deleted", "Member deleted successfully.")


    def clear_all_members():
        # Display a confirmation dialog
        confirmation = messagebox.askyesno("Confirm Clear",
                                           "Are you sure you want to clear all members? This action cannot be undone.")

        if confirmation:
            # Clear all data from the Excel file
            data_frame.drop(data_frame.index, inplace=True)
            data_frame.to_excel('faltedata.xlsx', index=False)

            # Update the table display after clearing data
            populate_table()

            # Show success alert
            messagebox.showinfo("All Members Cleared", "All members have been cleared successfully.")

    def export_excel_file():
        # Set the default file name
        default_file_name = "exportdata.xlsx"

        # Display a file save dialog with the default file name
        download_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")],
                                                     initialfile=default_file_name)

        # If the user didn't cancel the dialog
        if download_path:
            # Save the data frame to the selected path
            data_frame.to_excel(download_path, index=False)
            print(f"Excel file downloaded to: {download_path}")

    def add_member_from_main_gui():
        flate_number = flate_number_entry.get()
        vehicle = vehicle_entry.get()
        owner_name = number_plate_entry.get()
        number_plate = owner_name_entry.get()
        number = number_entry.get()

        # Check if the member already exists
        if any(data_frame['Number Plate'] == number_plate):
            messagebox.showinfo("Member Already Exists", "This member already exists in the database.")
            return

        # Append the data to the Excel file
        data_frame.loc[len(data_frame)] = [flate_number, vehicle, number_plate, owner_name, number]
        data_frame.to_excel('faltedata.xlsx', index=False)

        # Clear the input fields
        number_plate_entry.delete(0, tk.END)
        owner_name_entry.delete(0, tk.END)
        number_entry.delete(0, tk.END)
        flate_number_entry.delete(0, tk.END)
        vehicle_entry.delete(0, tk.END)

        # Show success alert
        messagebox.showinfo("Member Added", "Member added successfully.")

    # Create a frame for the buttons
    button_frame = tk.Frame(member_management_window)
    button_frame.pack()

    # Create a new frame for the "Clear All Members" button
    new_button_frame = tk.Frame(member_management_window)
    new_button_frame.pack()

    # Create a button to add the member from the main GUI
    add_button = tk.Button(button_frame, text="Add Member", command=add_member_from_main_gui)
    add_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create the "Edit Member" button
    edit_member_button = tk.Button(button_frame, text="Update Member", command=edit_member)
    edit_member_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create the "Delete Member" button
    delete_member_button = tk.Button(button_frame, text="Delete Member", command=delete_member)
    delete_member_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create a button to clear all members
    clear_all_button = tk.Button(button_frame, text="Clear All Members", command=clear_all_members)
    clear_all_button.pack(side=tk.LEFT, padx=10)

    # Create the download sample file
    download_button = tk.Button(new_button_frame, text="Download Excel File", command=download_excel_file)
    download_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create a button to  browse file
    browse_button = tk.Button(new_button_frame, text="Browse Excel File", command=browse_file)
    browse_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create a button to export the Excel file
    export_excel_button = tk.Button(new_button_frame, text="Export Excel File", command=export_excel_file)
    export_excel_button.pack(side=tk.LEFT, padx=10)

    ######see members
    # Create a table to display members' data
    members_table = ttk.Treeview(member_management_window,
                                 columns=("Flate Number", "Vehicle", "Number Plate", "Owner Name", "Number"))
    members_table.heading("Flate Number", text="Flate ")
    members_table.heading("Vehicle", text="Vehicle")
    members_table.heading("Number Plate", text="Number Plate")
    members_table.heading("Owner Name", text="Owner Name")
    members_table.heading("Number", text="Number")

    # Set the column widths
    members_table.column("#0", width=0, stretch=tk.NO)
    members_table.column("Flate Number", width=50, anchor="center")
    members_table.column("Vehicle", width=50, anchor="center")
    members_table.column("Number Plate", width=150, anchor="center")
    members_table.column("Owner Name", width=100, anchor="center")
    members_table.column("Number", width=100, anchor="center")

    for row in members_table.get_children():
        members_table.delete(row)

        # Fetch data from the Excel file and add to the table
    data_frame = pd.read_excel('faltedata.xlsx')
    for index, row in data_frame.iterrows():
        members_table.insert("", "end", values=(
        row['Flate Number'], row['Vehicle'], row['Number Plate'], row['Owner Name'], row['Number']))

        # Create a vertical scrollbar for the table
    members_scrollbar = ttk.Scrollbar(member_management_window, orient="vertical", command=members_table.yview)
    members_table.configure(yscrollcommand=members_scrollbar.set)

    # Pack the table and the scrollbar
    members_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
    members_scrollbar.pack(side=tk.RIGHT, fill="y", pady=5)


def update_filtered_table(filtered_data):
    # Clear the existing rows from the table
    for row in number_plates_table.get_children():
        number_plates_table.delete(row)

    # Populate the table with filtered data
    for item in filtered_data:
        number_plates_table.insert("", "end", values=(item[0], item[1], item[2], item[3], item[4]))


def clear_all_data():
    # Show a confirmation dialog
    confirmed = messagebox.askyesno("Confirmation", "Are you sure you want to clear all data?")

    if confirmed:
        # Clear the existing rows from the table and the image references
        number_plates_table.delete(*number_plates_table.get_children())
        number_plates_table.image_references = []

        # Clear the unique_valid_number_plates set to prevent further display
        unique_valid_number_plates.clear()

        # Clear data from the database
        cursor.execute("DELETE FROM number_plates")
        connection.commit()


# Initialize database connection (replace with your actual connection setup)
connection = sqlite3.connect("number_plate.db")
cursor = connection.cursor()


def open_filter_window():
    # Create a new window for filtering data
    filter_window = tk.Toplevel(root)
    filter_window.title("Filter Data")

    def process_filter():
        number_plate = number_plate_filter_entry.get()
        start_time = start_time_filter_entry.get()
        end_time = end_time_filter_entry.get()
        owner_name = owner_name_filter_entry.get()

        # Clear the existing rows from the table
        for row in filtered_data_table.get_children():
            filtered_data_table.delete(row)

        # Define the base SQL query for fetching data
        query = "SELECT * FROM number_plates WHERE 1=1"
        # query += " ORDER BY time DESC LIMIT 5"

        # Define placeholders and parameters list for the query
        params = []

        # Construct the query and parameters based on the provided filter criteria
        if number_plate:
            query += " AND number_plate LIKE ?"
            params.append(f"%{number_plate}%")
        if start_time and end_time:
            query += " AND time BETWEEN ? AND ?"
            params.extend([start_time, end_time])
        if owner_name:
            query += " AND owner_name LIKE ?"
            params.append(f"%{owner_name}%")

        # Fetch data from the database based on the constructed query and parameters
        cursor.execute(query, params)
        filtered_data = cursor.fetchall()

        # Display the filtered data in the table
        for item in filtered_data:
            filtered_data_table.insert("", "end",
                                       values=(item[0], item[2], item[3], item[4], item[5]))  # Adjust indexes if needed

    def export_filtered_data():
        number_plate = number_plate_filter_entry.get()
        start_time = start_time_filter_entry.get()
        end_time = end_time_filter_entry.get()
        owner_name = owner_name_filter_entry.get()
        query = "SELECT * FROM number_plates WHERE number_plate LIKE ? AND time BETWEEN ? AND ? AND owner_name LIKE ?"

        # Fetch the filtered data from the database
        cursor.execute(query, (f"%{number_plate}%", start_time, end_time, f"%{owner_name}%"))
        filtered_data = cursor.fetchall()

        # Create a Pandas DataFrame from the filtered data
        df = pd.DataFrame(filtered_data, columns=["ID", "IMAGE", "Number Plate", "Time", "Owner Name", "Number"])

        # Default file name and file type
        default_file_name = "filtered_data.xlsx"
        file_types = [("Excel files", "*.xlsx")]

        # Prompt the user to choose a file location for saving
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=file_types,
                                                 initialfile=default_file_name)

        # Export the DataFrame to an Excel file
        if file_path:
            df.to_excel(file_path, index=False)

    # Create input fields for Number Plate, Time, and Owner Name
    frame1 = ttk.Frame(filter_window)
    frame1.pack(padx=10, pady=10)

    number_plate_label = tk.Label(frame1, text="Number Plate :                  ")
    number_plate_filter_entry = tk.Entry(frame1)
    number_plate_label.pack(side=tk.LEFT)
    number_plate_filter_entry.pack(side=tk.LEFT)

    frame2 = ttk.Frame(filter_window)
    frame2.pack(padx=10, pady=10)

    start_time_label = tk.Label(frame2, text="From Date (dd-mm-yyyy):")
    start_time_filter_entry = tk.Entry(frame2)
    start_time_label.pack(side=tk.LEFT)
    start_time_filter_entry.pack(side=tk.LEFT)



    end_time_label = tk.Label(frame2, text="To Date (dd-mm-yyyy):   ")
    end_time_filter_entry = tk.Entry(frame2)
    end_time_label.pack(side=tk.LEFT)
    end_time_filter_entry.pack(side=tk.RIGHT)



    owner_name_label = tk.Label(frame1, text="Owner Name :                  ")
    owner_name_filter_entry = tk.Entry(frame1)
    owner_name_label.pack(side=tk.LEFT)
    owner_name_filter_entry.pack(side=tk.RIGHT)

    # Create a "Submit" button to process the filter

    frame4 = ttk.Frame(filter_window)
    frame4.pack(padx=10, pady=10)

    submit_button = tk.Button(frame4, text="Submit", command=process_filter)
    submit_button.pack(side=tk.LEFT, padx=10, pady=10)

    export_button = tk.Button(frame4, text="Export", command=export_filtered_data)
    export_button.pack(side=tk.RIGHT, padx=10, pady=10)

    # Create a table to display filtered data
    filtered_data_table = ttk.Treeview(filter_window, columns=("ID", "Number Plate", "Time", "Owner Name", "Number"))
    filtered_data_table.heading("ID", text="ID")
    filtered_data_table.heading("#0", text="ID")
    filtered_data_table.heading("Number Plate", text="Number Plate")
    filtered_data_table.heading("Time", text="Time")
    filtered_data_table.heading("Owner Name", text="Owner Name")
    filtered_data_table.heading("Number", text="Number")

    # Set the column widths
    filtered_data_table.column("ID", width=50, anchor="center")
    filtered_data_table.column("#0", width=0, stretch=tk.NO)
    filtered_data_table.column("Number Plate", width=100, anchor="center")
    filtered_data_table.column("Time", width=150, anchor="center")
    filtered_data_table.column("Owner Name", width=100, anchor="center")
    filtered_data_table.column("Number", width=100, anchor="center")

    # Create a vertical scrollbar for the table
    table_scrollbar = ttk.Scrollbar(filter_window, orient="vertical", command=filtered_data_table.yview)
    filtered_data_table.configure(yscrollcommand=table_scrollbar.set)

    # Pack the table and the scrollbar to the right side
    filtered_data_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
    table_scrollbar.pack(side=tk.RIGHT, fill="y", pady=5)


def populate_table():
    # Clear the existing rows
    for row in number_plates_table.get_children():
        number_plates_table.delete(row)

    # Fetch data from the database and add to the table
    cursor.execute("SELECT id, number_plate, time, owner_name, number FROM number_plates ORDER BY id DESC")
    data = cursor.fetchall()

    # Insert the latest scanned number plate at the top
    if data:
        latest_item = data[0]
        number_plate_image_path = f"{output_folder}/{latest_item[1]}.png"
        if os.path.exists(number_plate_image_path):
            plate_image = Image.open(number_plate_image_path)
            plate_image.thumbnail((100, 100))
            plate_image_tk = ImageTk.PhotoImage(plate_image)
        else:
            # If the image does not exist, use a placeholder image
            plate_image = Image.new('RGB', (50, 50), color='gray')
            plate_image_tk = ImageTk.PhotoImage(plate_image)

        number_plates_table.insert("", "end", values=(
        latest_item[0], latest_item[1], latest_item[2], latest_item[3], latest_item[4]), image=plate_image_tk)
        # Keep a reference to the image to prevent it from being garbage collected
        number_plates_table.image_references.append(plate_image_tk)

    # Insert the remaining data
    for item in data[1:]:
        number_plate_image_path = f"{output_folder}/{item[1]}.png"
        if os.path.exists(number_plate_image_path):
            plate_image = Image.open(number_plate_image_path)
            plate_image.thumbnail((100, 100))
            plate_image_tk = ImageTk.PhotoImage(plate_image)
        else:
            # If the image does not exist, use a placeholder image
            plate_image = Image.new('RGB', (50, 50), color='gray')
            plate_image_tk = ImageTk.PhotoImage(plate_image)

        number_plates_table.insert("", "end", values=(item[0], item[1], item[2], item[3], item[4]),
                                   image=plate_image_tk)
        # Keep a reference to the image to prevent it from being garbage collected
        number_plates_table.image_references.append(plate_image_tk)



# ##########   camera settings
# latest_camera_value = None
#
# conn = sqlite3.connect('camera_settings.db')
# cursor = conn.cursor()
# cursor.execute('''
#         CREATE TABLE IF NOT EXISTS camera_settings (
#             id INTEGER PRIMARY KEY,
#             camera_type TEXT,
#             camera_value INT,
#             gap_second INT
#         )
#     ''')
# cursor.execute('''
#     INSERT OR IGNORE INTO camera_settings (id, camera_type, camera_value, gap_second)
#     VALUES (1, 'default', 0, 60)
# ''')
# conn.commit()
# conn.close()
#
#
# def create_camera_settings_gui():
#     app = QApplication(sys.argv)
#     window = QMainWindow()
#     window.setWindowTitle("Camera Settings")
#     window.setGeometry(100, 100, 400, 300)
#
#     central_widget = QWidget()
#     window.setCentralWidget(central_widget)
#
#     layout = QVBoxLayout()
#
#     # Create a horizontal layout for the gap input and set button
#     gap_layout = QHBoxLayout()
#
#     gap_second_label = QLabel("Gap (in seconds):")
#     gap_second_edit = QLineEdit()
#     set_gap_button = QPushButton("Set Gap")
#
#     gap_layout.addWidget(gap_second_label)
#     gap_layout.addWidget(gap_second_edit)
#     gap_layout.addWidget(set_gap_button)
#
#     layout.addLayout(gap_layout)  # Add the horizontal layout to the main layout
#
#     camera_type_label = QLabel("Select Camera Type:")
#     camera_type_combobox = QComboBox()
#     camera_type_combobox.addItem("USB")
#     camera_type_combobox.addItem("RTSP")
#     camera_type_combobox.addItem("Webcam")
#
#     layout.addWidget(camera_type_label)
#     layout.addWidget(camera_type_combobox)
#
#     # Create a stacked widget to hold different settings widgets
#     stacked_widget = QStackedWidget()
#
#     # USB settings widget
#     usb_widget = QWidget()
#     usb_layout = QVBoxLayout()
#     usb_combo_label = QLabel("Select USB Camera Number:")
#     usb_combo = QComboBox()
#     for i in range(1, 11):
#         usb_combo.addItem(str(i))
#     usb_layout.addWidget(usb_combo_label)
#     usb_layout.addWidget(usb_combo)
#     usb_widget.setLayout(usb_layout)
#     stacked_widget.addWidget(usb_widget)
#
#     # RTSP settings widget
#     rtsp_widget = QWidget()
#     rtsp_layout = QVBoxLayout()
#     rtsp_link_label = QLabel("Enter RTSP Link:")
#     rtsp_link_edit = QLineEdit()
#     rtsp_layout.addWidget(rtsp_link_label)
#     rtsp_layout.addWidget(rtsp_link_edit)
#     rtsp_widget.setLayout(rtsp_layout)
#     stacked_widget.addWidget(rtsp_widget)
#
#     # Webcam settings widget (auto-adds 0)
#     webcam_widget = QWidget()
#     webcam_layout = QVBoxLayout()
#     webcam_label = QLabel("Camera Number: 0 (Webcam)")
#     webcam_widget.setLayout(webcam_layout)
#     stacked_widget.addWidget(webcam_widget)
#
#     layout.addWidget(stacked_widget)
#
#     save_button = QPushButton("Save Settings")
#     layout.addWidget(save_button)
#
#     central_widget.setLayout(layout)
#
#     conn = sqlite3.connect('camera_settings.db')
#     cursor = conn.cursor()
#
#     def save_settings():
#         global latest_camera_value
#         camera_type = camera_type_combobox.currentText()
#
#         if camera_type == "USB":
#             camera_value = usb_combo.currentText()
#         elif camera_type == "RTSP":
#             camera_value = rtsp_link_edit.text()
#         else:
#             camera_value = "0"  # Webcam
#
#         if len(camera_value) > 3:
#             camera_value = str(camera_value)
#         else:
#             try:
#                 camera_value = int(camera_value)
#             except ValueError:
#                 QMessageBox.warning(window, "Invalid Value",
#                                     "Camera value must be an integer when its length is less than or equal to 3.")
#                 return
#
#         # Retrieve the gap seconds value
#         gap_seconds = gap_second_edit.text()
#         try:
#             gap_seconds = int(gap_seconds)
#         except ValueError:
#             gap_seconds = 0  # Default to 0 if the input is not a valid integer
#
#         # Store the settings in the database
#         cursor.execute("UPDATE camera_settings SET camera_type=?, camera_value=?, gap_second=? WHERE id=1",
#                        (camera_type, camera_value, gap_seconds))
#         conn.commit()
#         QMessageBox.information(window, "Success", "Settings saved successfully!")
#
#         # Update the latest_camera_value
#         latest_camera_value = camera_value
#         print("Latest Camera Value:", latest_camera_value)
#
#         # Close the settings window after saving
#         window.close()
#
#     def camera_type_changed(index):
#         stacked_widget.setCurrentIndex(index)
#
#     def set_gap_seconds():
#         gap_text = gap_second_edit.text()
#         try:
#             gap_seconds = int(gap_text)
#             # You can perform additional validation here if needed
#             # Update the gap_second value in the database
#             cursor.execute("UPDATE camera_settings SET gap_second=? WHERE id=1", (gap_seconds,))
#             conn.commit()
#             QMessageBox.information(window, "Success", f"Gap set to {gap_seconds} seconds!")
#         except ValueError:
#             QMessageBox.warning(window, "Invalid Value", "Gap must be an integer.")
#
#     camera_type_combobox.currentIndexChanged.connect(camera_type_changed)
#     save_button.clicked.connect(save_settings)
#     set_gap_button.clicked.connect(set_gap_seconds)
#
#     window.show()
#     sys.exit(app.exec_())
#
#
#
# # Retrieve the latest camera value and gap seconds from the database
# conn = sqlite3.connect('camera_settings.db')
# cursor = conn.cursor()
# cursor.execute("SELECT camera_value, gap_second FROM camera_settings")
# existing_settings = cursor.fetchone()
#
# if existing_settings:
#     latest_camera_value, latest_gap_seconds = existing_settings  # Extract values from the tuple
# else:
#     latest_camera_value = None
#     latest_gap_seconds = 0  # Default gap value if not found
#
# # Set the 'cap' variable with the latest camera value
# cap = cv2.VideoCapture(latest_camera_value) if latest_camera_value is not None else cv2.VideoCapture(0)
# # Check if the video capture was successfully opened
# if not cap.isOpened():
#     print("Error: Camera or video source not found or cannot be opened.")
#     # sys.exit(1)  # Exit the program with an error code
# print("Latest Camera Value:", latest_camera_value)
# print("Latest Gap Seconds:", latest_gap_seconds)
#





# video_path = 'Reverse_Parking.mp4'
# # Create a VideoCapture object to read the input video
# cap = cv2.VideoCapture(video_path)



# # Create a window with the specified size
# cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Video', window_width, window_height)

# Initialize webcam capture
# cap = cv2.VideoCapture("rtsp://AdminBonrixx:Admin@1234@192.168.29.70:554/stream1")
# cap = cv2.VideoCapture(0)
# cap = cv2.imread('num 2.jpeg')
# cap = cv2.VideoCapture(int(latest_camera_value) if latest_camera_value is not None else 0)


# ADD #UPDATE #AND # DELETE

# Set your Tesseract OCR path here
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QStackedWidget, QMessageBox, QFileDialog
import sys
import sqlite3
import cv2

# Initialize the latest_camera_value and latest_gap_seconds with default values
latest_camera_value = None
latest_gap_seconds = 0

##########   camera settings

conn = sqlite3.connect('camera_settings.db')
cursor = conn.cursor()
cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_settings (
            id INTEGER PRIMARY KEY,
            camera_type TEXT,
            camera_value TEXT,
            gap_second INT
        )
    ''')
cursor.execute('''
    INSERT OR IGNORE INTO camera_settings (id, camera_type, camera_value, gap_second)
    VALUES (1, 'default', '0', 60)
''')
conn.commit()
conn.close()

def create_camera_settings_gui():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Camera Settings")
    window.setGeometry(100, 100, 400, 300)

    central_widget = QWidget()
    window.setCentralWidget(central_widget)

    layout = QVBoxLayout()

    # Create a horizontal layout for the gap input and set button
    gap_layout = QHBoxLayout()

    gap_second_label = QLabel("Gap (in seconds):")
    gap_second_edit = QLineEdit()
    set_gap_button = QPushButton("Set Gap")

    gap_layout.addWidget(gap_second_label)
    gap_layout.addWidget(gap_second_edit)
    gap_layout.addWidget(set_gap_button)

    layout.addLayout(gap_layout)  # Add the horizontal layout to the main layout

    camera_type_label = QLabel("Select Camera Type:")
    camera_type_combobox = QComboBox()
    camera_type_combobox.addItem("USB")
    camera_type_combobox.addItem("RTSP")
    camera_type_combobox.addItem("Webcam")
    camera_type_combobox.addItem("File")  # Add "File" option

    layout.addWidget(camera_type_label)
    layout.addWidget(camera_type_combobox)

    # Create a stacked widget to hold different settings widgets
    stacked_widget = QStackedWidget()

    # USB settings widget
    usb_widget = QWidget()
    usb_layout = QVBoxLayout()
    usb_combo_label = QLabel("Select USB Camera Number:")
    usb_combo = QComboBox()
    for i in range(1, 11):
        usb_combo.addItem(str(i))
    usb_layout.addWidget(usb_combo_label)
    usb_layout.addWidget(usb_combo)
    usb_widget.setLayout(usb_layout)
    stacked_widget.addWidget(usb_widget)

    # RTSP settings widget
    rtsp_widget = QWidget()
    rtsp_layout = QVBoxLayout()
    rtsp_link_label = QLabel("Enter RTSP Link:")
    rtsp_link_edit = QLineEdit()
    rtsp_layout.addWidget(rtsp_link_label)
    rtsp_layout.addWidget(rtsp_link_edit)
    rtsp_widget.setLayout(rtsp_layout)
    stacked_widget.addWidget(rtsp_widget)

    # Webcam settings widget (auto-adds 0)
    webcam_widget = QWidget()
    webcam_layout = QVBoxLayout()
    webcam_label = QLabel("Camera Number: 0 (Webcam)")
    webcam_widget.setLayout(webcam_layout)
    stacked_widget.addWidget(webcam_widget)

    # File settings widget
    file_widget = QWidget()
    file_layout = QVBoxLayout()
    file_button = QPushButton("Browse Video File")
    file_label = QLabel("Selected Video File:")
    file_name_label = QLabel("No file selected")

    def browse_file():
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(window, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        if file_path:
            file_name_label.setText(file_path)

    file_button.clicked.connect(browse_file)
    file_layout.addWidget(file_button)
    file_layout.addWidget(file_label)
    file_layout.addWidget(file_name_label)
    file_widget.setLayout(file_layout)
    stacked_widget.addWidget(file_widget)

    layout.addWidget(stacked_widget)

    save_button = QPushButton("Save Settings")
    layout.addWidget(save_button)

    central_widget.setLayout(layout)

    conn = sqlite3.connect('camera_settings.db')
    cursor = conn.cursor()

    def save_settings():
        global latest_camera_value
        camera_type = camera_type_combobox.currentText()

        if camera_type == "USB":
            camera_value = usb_combo.currentText()
        elif camera_type == "RTSP":
            camera_value = rtsp_link_edit.text()
        elif camera_type == "File":
            camera_value = file_name_label.text()
        else:
            camera_value = "0"  # Webcam

        if len(camera_value) > 3:
            camera_value = str(camera_value)
        else:
            try:
                camera_value = int(camera_value)
            except ValueError:
                QMessageBox.warning(window, "Invalid Value",
                                    "Camera value must be an integer when its length is less than or equal to 3.")
                return

        # Retrieve the gap seconds value
        gap_seconds = gap_second_edit.text()
        try:
            gap_seconds = int(gap_seconds)
        except ValueError:
            gap_seconds = 0  # Default to 0 if the input is not a valid integer

        # Store the settings in the database
        cursor.execute("UPDATE camera_settings SET camera_type=?, camera_value=?, gap_second=? WHERE id=1",
                       (camera_type, camera_value, gap_seconds))
        conn.commit()
        QMessageBox.information(window, "Success", "Settings saved successfully!")

        # Update the latest_camera_value
        latest_camera_value = camera_value
        print("Latest Camera Value:", latest_camera_value)

        # Close the settings window after saving
        window.close()

    def camera_type_changed(index):
        stacked_widget.setCurrentIndex(index)

    def set_gap_seconds():
        gap_text = gap_second_edit.text()
        try:
            gap_seconds = int(gap_text)
            # You can perform additional validation here if needed
            # Update the gap_second value in the database
            cursor.execute("UPDATE camera_settings SET gap_second=? WHERE id=1", (gap_seconds,))
            conn.commit()
            QMessageBox.information(window, "Success", f"Gap set to {gap_seconds} seconds!")
        except ValueError:
            QMessageBox.warning(window, "Invalid Value", "Gap must be an integer.")

    camera_type_combobox.currentIndexChanged.connect(camera_type_changed)
    save_button.clicked.connect(save_settings)
    set_gap_button.clicked.connect(set_gap_seconds)

    window.show()
    sys.exit(app.exec_())

# ...

# Retrieve the latest camera value and gap seconds from the database
conn = sqlite3.connect('camera_settings.db')
cursor = conn.cursor()
cursor.execute("SELECT camera_value, gap_second FROM camera_settings")
existing_settings = cursor.fetchone()

if existing_settings:
    latest_camera_value, latest_gap_seconds = existing_settings  # Extract values from the tuple
else:
    latest_camera_value = None
    latest_gap_seconds = 0  # Default gap value if not found

# Set the 'cap' variable with the latest camera value
cap = cv2.VideoCapture(latest_camera_value) if latest_camera_value is not None else cv2.VideoCapture(0)
# Check if the video capture was successfully opened
if not cap.isOpened():
    print("Error: Camera or video source not found or cannot be opened.")
    # sys.exit(1)  # Exit the program with an error code
print("Latest Camera Value:", latest_camera_value)
print("Latest Gap Seconds:", latest_gap_seconds)









pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Database setup
connection = sqlite3.connect("number_plate.db")
cursor = connection.cursor()



# Create the 'number_plates' table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS number_plates (
        id INTEGER PRIMARY KEY,
        img TEXT,
        number_plate TEXT,
        time DATETIME,
        owner_name TEXT,
        number TEXT
    )
""")

# Commit the changes
connection.commit()
# # Fetch data from the database and add to the table
# cursor.execute("SELECT id, number_plate, time, owner_name, number FROM number_plates ORDER BY id DESC")
# data = cursor.fetchall()


# Check if the 'number' column exists in the table
cursor.execute('PRAGMA table_info("number_plates")')
columns = cursor.fetchall()
if not any(column[1] == 'number' for column in columns):
    # Add the 'number' column to the table if it doesn't exist
    cursor.execute("ALTER TABLE number_plates ADD COLUMN number TEXT")
    connection.commit()

# Read the data from the Excel file
data_frame = pd.read_excel('faltedata.xlsx')

# Create a dictionary to map Number Plate to Owner Name and Number
# number_plate_mapping = dict(zip(data_frame['Number Plate'], data_frame[['flat number','Owner Name', 'Number']].values))
number_plate_mapping = dict(
    zip(data_frame['Number Plate'], data_frame[['Owner Name', 'Number', 'Flate Number', 'Vehicle']].values))


def display_data():
    # Connect to the database
    conn = sqlite3.connect('number_plate.db')
    cursor = conn.cursor()

    # Fetch data from the database
    cursor.execute('SELECT * FROM number_plates')
    data = cursor.fetchall()

    # Update the table with the retrieved data
    for item in data:
        number_plates_table.insert("", "end", values=(item[0], item[2], item[3], item[4], item[5]))

    # Close the database connection
    conn.close()


root = tk.Tk()
root.title("Number Plate Recognition")

# Create a frame to hold the webcam and the table
main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=10)

# Create a label to display the video stream
video_label = tk.Label(main_frame)
video_label.pack(side=tk.LEFT, padx=5)

# Create a table to display saved number plates
number_plates_table = ttk.Treeview(main_frame, columns=("Image", "Number Plate", "Time", "Owner Name", "Number"))
number_plates_table.heading("Image", text="ID")
number_plates_table.heading("#0", text="Image")
number_plates_table.heading("Number Plate", text="Number Plate")
number_plates_table.heading("Time", text="Time")
number_plates_table.heading("Owner Name", text="Owner Name")
number_plates_table.heading("Number", text="Number")

# Set the column widths
number_plates_table.column("Image", width=50, anchor="center")
number_plates_table.column("#0", width=150, anchor="center")
number_plates_table.column("Number Plate", width=100, anchor="center")
number_plates_table.column("Time", width=150, anchor="center")
number_plates_table.column("Owner Name", width=100, anchor="center")
number_plates_table.column("Number", width=100, anchor="center")
#


# Create a vertical scrollbar for the table
table_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=number_plates_table.yview)
number_plates_table.configure(yscrollcommand=table_scrollbar.set)

# Pack the table and the scrollbar to the right side
number_plates_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
# table_scrollbar.pack(side=tk.RIGHT, fill="y", pady=5)

# Keep references to the images displayed in the table
number_plates_table.image_references = []

# Directly display the data when the GUI is started
display_data()

# Create a frame for the webcam display and the table

content_frame = ttk.Frame(main_frame)
content_frame.pack(side=tk.TOP, padx=5, pady=5)
#
# # Create a label to display the video stream
# Define the desired width and height for the label

label_width = 800  # Set your desired width
label_height = 600  # Set your desired height

# # Create a label to display the video stream with the specified size
video_label = tk.Label(content_frame, width=label_width, height=label_height)
video_label.pack(side=tk.LEFT, padx=5)

# video_label = tk.Label(content_frame)
# video_label.pack(side=tk.LEFT, padx=5)

# Create a vertical scrollbar for the table
table_scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=number_plates_table.yview)
number_plates_table.configure(yscrollcommand=table_scrollbar.set)

# Pack the table and the scrollbar to the right side    CAMERA AND TABLE
number_plates_table.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
table_scrollbar.pack(side=tk.RIGHT, fill="y", pady=5)

# Create a "Member Management" button
member_management_button = tk.Button(main_frame, text="Member Management", command=open_member_management)
member_management_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create the "Clear All Data" button
clear_button = tk.Button(main_frame, text="Clear All Data", command=clear_all_data)
clear_button.pack(side=tk.LEFT, padx=5, pady=5)

# # Add a "Filter Data" button to the main GUI
filter_button = tk.Button(main_frame, text="Filter Data", command=open_filter_window)
filter_button.pack(side=tk.LEFT, padx=10, pady=10)

# # Create a button to open settings window
open_settings_button = tk.Button(main_frame, text="Settings", command=create_camera_settings_gui)
open_settings_button.pack(side=tk.LEFT, padx=10, pady=10)


def update_log_file():
    try:
        with open("my_log_file.log", "r") as log_file:  # Replace with your file name
            log_lines = log_file.readlines()

            # Keep only the latest 5000 lines
            log_lines = log_lines[-5000:]
            # Reverse the order of log_lines
            log_lines.reverse()
            log_text.delete(1.0, tk.END)  # Clear the existing text
            log_text.insert(1.0, "".join(log_lines))  # Insert log lines at the beginning
    except FileNotFoundError:
        log_text.delete(1.0, tk.END)
        log_text.insert(tk.END, "Log file not found.")


def update_periodically():
    update_log_file()
    root.after(5000, update_periodically)


logframe = tk.Frame(root)
logframe.pack()

log_text = tk.Text(logframe, wrap=tk.WORD, width=170, height=10)
scrollbar = tk.Scrollbar(logframe, command=log_text.yview)
log_text.config(yscrollcommand=scrollbar.set)

log_text.pack(side=tk.LEFT)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

update_log_file()
update_periodically()

update_video_stream()

# Run the GUI main event loop
root.mainloop()
#
# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Write unique valid number plates to a text file
write_unique_plates_to_file()

# Don't forget to close the database connection after the GUI is closed.
cursor.close()
connection.close()





#rtsp://AdminBonrixx:Admin@1234@192.168.29.70:554/stream1





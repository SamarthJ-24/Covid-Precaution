{%load static %}
<html>
	<head>
		<link rel="stylesheet" href="{% static "css/main.css" %}">
	</head>	

<body>
<div class='sidebar'>
		<div class='sidebar-area'> 
			<div class='row' style='margin-bottom: 20px;'> 
				<div class='col-md-6'> 
					<div class='user-profile'> 
						<img src='images/3678412 - doctor medical care medical help stethoscope.png' class='img-responsive' style='max-height: 80px;' /> 
					</div>
				</div> 
				<div class='col-md-6'> 
					<div class='user-names'> 
					
					</div>
					
					<div class='user-role'> 
						
					</div>
				</div> 
			</div> 
			<ul class='sidebar-menu'>
				<li><a href='index.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_account_balance_wallet_white_24dp.png' %}" /> Dashboard</a></li>
				<li><a href='profile.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_account_box_white_24dp.png' %}" /> Profile</a></li>
				<li><a href='new-patient.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_group_add_white.png' %}" /> Add Patient</a></li>
				<li><a href='patients.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_assignment_ind_white_24dp.png' %}" /> Patients Book</a></li>
				<li><a href='add-doctors.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_group_add_white.png' %}" /> Add Doctors</a></li>
				<li><a href='doctors-record.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_group_add_white.png'%}"  /> Doctors' Records</a></li>
				<li><a href='appointments.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_alarm_white_24dp.png'%}"  /> Appointment</a></li>
				<li><a href='http://localhost:5000/'><img class='sidebar-menu-icon' src="{% static 'images/ic_group_work_white_24dp.png'%}"  /> Face Recognition and Mask Detection</a></li>
				<li><a href='outbreaks.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_group_work_white_24dp.png'%}"  /> Social Distancing Monitoring</a></li>
				<li><a href='hiv.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_error_outline_white_24dp.png'%}"  /> Covid-19</a></li>
				<li><a href='reports.php'><img class='sidebar-menu-icon' src="{% static 'images/ic_receipt_white_24dp.png'%}"  /> Covid-19 Reports</a></li>
				
			</ul> 
		</div> 
	</div>
</body>
<html>
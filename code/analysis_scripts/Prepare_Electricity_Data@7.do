************************
*Electricity Monitoring*
************************

//Preable
	version 13
	set more off
	clear all
	
//Set versions
	loc vers = 3
	loc vers2 = 4
	loc j = 11 //box list
	
//Set file location
	*global path "C:\Users\wb255520\Box Sync\Listening2Tajikistan\01.Eletricity_monitoring\"					/* JPA 	*/
	global path "C:\Users\WB454594\Box Sync\Tajikistan\Listening2Tajikistan\01.Eletricity_monitoring\"		/* WS	*/
	*global path "C:\Users\WB472222\Box Sync\Listening2Tajikistan\01.Eletricity_monitoring\"		/* OKI	*/
	global root "$path\01.data"

//Open with import excel
	import excel "$root\sms.xlsx", clear firstrow
	
//Do some data cleaning
	drop toa ///
		sc_toa ///
		read	///
		status	///
		locked ///
		type	///
		subject ///
		backup_date	protocol ///
		date	///
		type ///
		contact_name ///
		service_center ///
		backup_set ///

//Clean variables needed
	split readable_date, g(d_t) p(" ")
	
//Clean incorrect date information
	g length_var = length(readable_date)
	g fix = 0
	replace fix = 1 if length_var!=19
	
	egen fix_nums = sieve(readable_date), keep(numeric)
	
	tostring fix_nums, replace
	g day_fix = substr(fix_nums, 1, 2)
	g new_fixdate = "12/" + day_fix + "/2016"
	g time_fix1 = substr(fix_nums, 7, 2)
	g time_fix2 = substr(fix_nums, 9, 2)
	g time_fix3 = substr(fix_nums, 11, 2)
	
	forvalues i = 1/3 {
		tostring time_fix`i', replace
		}
	
	g d__t5 = time_fix1 + ":" + time_fix2 + ":" + time_fix3
	
	*g fixdate = ""
	*replace fixdate = "12/" + d_t1 + "/2016" if fix == 1

	g fixdatetime = ""
	replace fixdatetime = new_fixdate + " " + d_t5
	br if fix==1
	
	
	replace readable_date = fixdatetime if fix == 1
	replace d_t1 = new_fixdate if fix == 1
	
//Set date and time	when message was recieved
	g double datetime = clock(readable_date, "MDYhms")
	format datetime %tc
	g date = date(d_t1, "MDY")
	format date %td
	rename d_t2 time_str
	
	gen phone_number = subinstr(address,"+","",.)
	destring phone_number, replace force
	format phone_number %12.0f
	
//Set date and time when message was sent
	generate double sent_datetime = date_sent + mdyhms(1,1,1970,5,0,0)
	format  sent_datetime %tcNN/DD/CCYY_HH:MM:SS
	
//Main indicators of interest
	split body, g(message) p("|" ":")
	
//Box id
	egen device_id = sieve(body), keep(numeric)
	destring device_id, replace
	
//Identify purpose of message
	g pback = regexm(body, "Power Back detected")
	g pfail = regexm(body, "Power Failure detected")
	g pfail_mon = regexm(body, "Monitoring... Power Failure")
	g pon_mon = regexm(body, "Monitoring... Power OK")
	g test = regexm(body, "Test SMS")
	replace test =1 if message1=="Test"
	g alive = 0
	replace alive = 1 if pfail_mon == 1
	replace alive = 1 if pon_mon == 1
	
	g pback2 = regexm(body, "Power Back.")
	replace pback = 1 if pback2==1 & test==0
	
	g pfail2 = regexm(body, "Power Failure. ID 1043")
	g pfail3 = regexm(body, "Power Failure. ID 1281")
	replace pfail = 1 if pfail2 == 1
	replace pfail = 1 if pfail3 == 1
	
//Replace dummies for tests with "0"
	loc idvars pback ///
		pfail ///
		pfail_mon ///
		pon_mon ///
		
	foreach i of loc idvars {
		replace `i' = 0 if test == 1
		}		

//drop incorrect message
/*
	drop if body=="Муштарии Мухтарам, Дархости шумо тахти №461640 пушида шуд."
	drop if body=="Муштари занги Шуморо интизор аст. Абонент просит Вас перезвонить"

//Perform check that all messages positively identified
	egen totdums = rowtotal(pback pfail pfail_mon pon_mon test)
	assert totdums != 0
	assert totdums != .
	assert totdums < 2 
	drop totdums
	
	*/
//temp
	egen totdums = rowtotal(pback pfail pfail_mon pon_mon test)
	drop if totdums == 0
	drop totdums
	
//Drop test data
	drop if test==1
	
//Create category variable
	g message_type = .
	replace message_type = 1 if pback == 1
	replace message_type = 2 if pfail == 1
	replace message_type = 3 if pfail_mon  == 1
	replace message_type = 4 if pon_mon == 1
	
	label de msgty	1 "Power Back" ///
				2 "Power Failure" ////
				3 "Power Failure Monitoring" ///
				4 "Power OK Monitoring"
		
	label val  message_type msgty
	
//Clean
	drop test ///
		pback2	 ///
		pfail2  ///
		pfail3 ///
		message1 ///
		message2 ///
		d_t1 ///
		count ///
		d_t3  ///
		d_t4  ///
		d_t5  ///
		fix  ///
		fixdate  ///
		fixdatetime 
		
//Find and clean duplicated texts
	duplicates drop device_id body readable_date, force
	duplicates tag device_id datetime, g(dupid)
	assert dupid == 0
	drop dupid
	
	label var	date "Stata Encoded Date Variable"
	label var 	readable_date "SMS Generated Time and Date Variable"
	label var 	datetime "Stata Encoded Date and Time Variable: Received"
	label var   sent_datetime "Stata Encoded Date and Time Variable: Sent"
	label var 	message_type "Processed Message Content"
	label var 	time_str "SMS Generated Time-of-Day"
	label var 	device_id "Monitoring Device Identifier"
	label var 	pback	"Power Back"
	label var 	pfail "Power Failure"
	label var 	pfail_mon "Monitoring: Power is Off"
	label var 	pon_mon "Monitoring: Power is On"
	label var 	body "Raw Body of SMS Message"
	label var 	phone "Phone Number Sending Text"
	
//Survival/Event Stata
	tsset device_id datetime
	sort device_id alive datetime
	g elapsed = datetime-datetime[_n-1]
	replace elapsed = elapsed/1000
	*g elapsed_min = elapsed/60
	replace elapsed = . if alive==1
	bys device_id alive: replace elapsed = . if _n==1
	
//Create difference sent/vs/ recieved
	g delay = (datetime-sent_datetime)
	replace delay = 1000 if delay==1 
	replace delay = round(delay/1000)
	label var delay "Seconds between sms sent and received" 
	
//2 different variables: one for duration of power on, another for time off
	g elapsed_fail = elapsed if pback==1
	g elapsed_on = elapsed if pfail==1
	
	sort device_id datetime
	la var elapsed "Time Elapsed Between Failure/Resume in Seconds"
	la var elapsed_fail "Time Elapsed: Spell of Power Failure in Seconds"
	la var elapsed_on "Time Elapsed: Spell of Power ON in Seconds"

//Clean and Label
	order date  ///
		readable_date  ///
		datetime  ///
		sent_datetime ///
		elapsed ///
		delay ///
		message_type ///
		phone ///
		time_str	  ///
		device_id	  ///
		pback	  ///
		pfail	  ///
		pfail_mon	  ///
		pon_mon	  ///
		alive ///
		body ///
		
//Create temporary file (to merge date of start)
	tempfile fullsms
	save `fullsms'
	
//Open list of boxes
	import excel "$root\Distribution_Boxes@`j'.xlsx", firstrow clear
	clonevar PSU_ID = ClusterId
	
//Merge coordinates
	merge m:1 PSU_ID using "$root\PSU_Coordinates"
	drop if _m==2
	drop _m
	format Phonenumber %12.0f

//Fix dates in wrong format
	replace DateCollectionStart = date("01Nov2016", "DMY") if BoxID==1106	
	replace DateCollectionStart = date("02Dec2016", "DMY") if BoxID==1093	

//Find and clean duplicated BoxIDs
	duplicates tag BoxID, g(dupid)
	assert dupid==0
	drop dupid

	sort BoxID 
	
	save "$root\Boxes", replace

	export excel using "$root\Clean_Full_Elect_L2T@`vers'.xlsx", firstrow(variables) sh(BoxDetails)  sheetrep
	export delimited using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\BoxDetails.csv", replace
	export delimited using "$root\BoxDetails.csv", replace

	drop hhidWASH	///
		Name	///
		Phonenumber

	order BoxID
		
	export excel using "$root\Clean_Full_Elect_L2T@`vers'.xlsx", firstrow(variables) sh(Boxes)  sheetrep
	export excel using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\Clean_Full_Elect_L2T@`vers'.xlsx", firstrow(variables) sh(Boxes)  sheetrep

	export excel using "$root\Clean_Full_Elect_L2T@`vers2'.xlsx", firstrow(variables) sh(Boxes)  sheetrep
	export excel using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\Clean_Full_Elect_L2T@`vers2'.xlsx", firstrow(variables) sh(Boxes)  sheetrep
	
	export delimited using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\Boxes.csv", replace
	export delimited using "$root\Boxes.csv", replace

	keep BoxID DateCollectionStart

//Save date start
	tempfile datestart
	save `datestart'
	
//Open sms data
	use `fullsms', clear
	clonevar BoxID = device_id 
	merge m:1 BoxID using `datestart'

	/*
	g checkdate = mdy(11, 22, 2016)
	format checkdate %td
	keep if _m==1  & date>checkdate
	*/

	drop if _m==2
	drop _m
	
	g delivered = 0
	replace delivered =1 if DateCollectionStart<=date
	
//Create "full" dataset
	save "$root\Clean_Full_Elect_L2T@`vers'.dta", replace
	export excel using "$root\Clean_Full_Elect_L2T@`vers'.xlsx", firstrow(variables) sh(Data) sheetrep
	export excel using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\Clean_Full_Elect_L2T@`vers'.xlsx", firstrow(variables) sh(Data) sheetrep

	export delimited using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\Data.csv", replace
	export delimited using "$root\Data.csv", replace
	
**************************************
// Prepare databased to report power outages
**************************************
/** Generage values indicating that at least one SMS was received */
**************************************
cd "$root\"
use "$root\Clean_Full_Elect_L2T@`vers'.dta", clear

drop if delivered == 0

drop if message_type == 1 | message_type == 2

sort BoxID datetime

keep BoxID datetime message_type DateCollectionStart

gen double  power_failure = datetime[_n]

foreach var in power_failure {
	gen double seconds = ss(`var')
	gen double `var'3 = `var'- seconds*999
	drop `var'
	rename `var'3 `var'
	format `var' %tc
	drop seconds
}

format power_failure  	%tc
format datetime         %tc

gen checks = 1

order BoxID message_type datetime power_failure 

gen date_powerfailure = dofc( power_failure)
gen date_powerfailure_hour = hh(power_failure)

collapse (sum) checks, by(BoxID date_powerfailure DateCollectionStart)

fillin BoxID date_powerfailure 

gsort BoxID -DateCollectionStart

bysort BoxID : replace DateCollectionStart = DateCollectionStart[_n-1] if DateCollectionStart == .

gen delivered = 1 if DateCollectionStart < date_powerfailure 

keep if delivered == 1

sort BoxID date_powerfailure 

recode check .=0

save check, replace

**************************************
/* Generate Expanded database per hour */
**************************************

use Clean_Full_Elect_L2T@3, clear

drop if delivered == 0

keep if message_type == 1 | message_type == 2

sort BoxID datetime

bysort BoxID : gen duration = minutes(datetime[_n+1]- datetime[_n]) if message_type[_n+1] == 1 & message_type[_n] == 2

keep BoxID datetime message_type duration DateCollectionStart

gen double  power_back = datetime[_n+1]
gen double  power_failure = datetime[_n]
gen double  datetimeorig = datetime

foreach var in power_back power_failure datetimeorig {
	gen double seconds = ss(`var')
	gen double `var'3 = `var'- seconds*999
	drop `var'
	rename `var'3 `var'
	format `var' %tc
	drop seconds
}

format power_back 		%tc
format power_failure  	%tc
format datetime         %tc

gen link = 1

keep if duration != .

order BoxID message_type datetime datetimeorig power_failure power_back duration DateCollectionStart

gen date_powerfailure = dofc( power_failure)
gen date_powerback = dofc( power_back)

gen date_powerfailure_hour = hh(power_failure)
gen date_powerback_hour = hh(power_back)

gen link2 = link

collapse (sum) link  (mean) link2 (sum) duration, by(BoxID date_powerfailure date_powerfailure_hour date_powerback  date_powerback_hour DateCollectionStart)

***********

gsort BoxID -DateCollectionStart

bysort BoxID : replace DateCollectionStart = DateCollectionStart[_n-1] if DateCollectionStart == .

gen delivered = 1 if DateCollectionStart < date_powerfailure 

keep if delivered == 1

***********

fillin BoxID date_powerfailure date_powerfailure_hour 

order BoxID date_powerfailure date_powerfailure_hour date_powerback date_powerback_hour

sort BoxID date_powerfailure date_powerfailure_hour 

preserve
	keep BoxID date_powerback date_powerback_hour _fillin
	drop if _fillin == 1
	gen date_powerfailure  		= date_powerback
	gen date_powerfailure_hour 	= date_powerback_hour
	sort BoxID date_powerfailure date_powerfailure_hour 
	save tmp, replace
restore

merge BoxID date_powerfailure date_powerfailure_hour  using tmp

sort BoxID date_powerfailure date_powerfailure_hour date_powerback date_powerback_hour

replace _merge = . if _merge == 1 & date_powerback == .

sort BoxID date_powerfailure date_powerfailure_hour date_powerback date_powerback_hour
replace _merge = 1 if _merge[_n-1]==1 & _merge[_n] ==. & BoxID[_n]==BoxID[_n-1] 

gen merge = _merge

recode _merge 3=1

sort BoxID date_powerfailure date_powerfailure_hour date_powerback date_powerback_hour

replace _merge = 3 if _merge[_n-1] == 3 & _merge[_n] == . & BoxID[_n]==BoxID[_n-1] & _merge[_n] != 1 

recode _merge 3=1 .=0

drop merge

rename _merge POWERout

drop link link2

format %td date_powerfailure date_powerback

gen double dhms = dhms( date_powerfailure , date_powerfailure_hour ,0,0)
format %tc dhms

sort BoxID date_powerfailure 

merge BoxID date_powerfailure  using check

recode check .=-9

********************************************
/* Save final files */
********************************************

save "$root\powerout", replace

export delimited using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\powerout.csv", replace
export delimited using "$root\powerout.csv", replace

export excel using "\\wbmstab04\Excel Share\GPVDR\ECA\TJK\Clean_Full_Elect_L2T@`vers2'.xlsx", sheet("powerout") sheetreplace firstrow(variables)
export excel using "$root\Clean_Full_Elect_L2T@`vers2'.xlsx", sheet("powerout") sheetreplace firstrow(variables)


	

*===================================================
* Newey West
*===================================================
* Use ff.csv
clear all
import delimited using ff.csv, clear
sort date
tsset date

capt frame drop neweyresult
frame create neweyresult lag stderr

foreach lag in 1 2 3 5 10 15 20 {
    newey mktrf, lag(`lag')
    loc a = sqrt(e(V)[1,1])
    frame post neweyresult (`lag') (`a')
}

frame neweyresult: save ff_mtkrf_newey.dta, replace

*===================================================
* Estimate beta
*===================================================
* Use xsstktestdata.csv
import delimited using xsstktestdata.csv, clear
tostring date, gen(datestr)
gen datemn = mofd(date(datestr, "YMD"))
format datemn %tm
keep permno datemn exret

capt frame drop ff
frame create ff
frame ff {
    import delimited ff.csv, clear
    tostring date, gen(datestr)
    gen datemn = mofd(date(datestr, "YMD"))
    format datemn %tm
}
frlink m:1 date, frame(ff)
frget mktrf smb hml umd, from(ff)
sort permno datemn

preserve
    rangestat (reg) exret mktrf smb hml umd, interval(datemn -24 -1) by(permno)
    keep if reg_nobs >= 6 & !missing(reg_nobs)
    drop se_* reg_* b_cons
    keep permno datemn b_mktrf b_smb b_hml b_umd
    keep if datemn >= mofd(mdy(1,1,2019))
    save testbetares1.dta, replace
restore

preserve
    rangestat (reg) exret mktrf smb hml umd, interval(datemn -20 -5) by(permno)
    keep if reg_nobs >= 10 & !missing(reg_nobs)
    drop se_* reg_* b_cons
    keep permno datemn b_mktrf b_smb b_hml b_umd
    keep if datemn >= mofd(mdy(9,1,2018))
    save testbetares2.dta, replace
restore

preserve
    rangestat (reg) exret mktrf smb hml umd, interval(datemn -12 0) by(permno)
    keep if reg_nobs >= 13 & !missing(reg_nobs)
    drop se_* reg_* b_cons
    keep permno datemn b_mktrf b_smb b_hml b_umd
    keep if datemn >= mofd(mdy(1,1,2018))
    save testbetares3.dta, replace
restore

preserve
    rangestat (reg) exret mktrf, interval(datemn -20 -5) by(permno)
    rename reg_nobs m1_nobs
    drop se_* reg_* b_cons
    rename b_* m1_*

    rangestat (reg) exret mktrf smb hml, interval(datemn -20 -5) by(permno)
    rename reg_nobs m2_nobs
    drop se_* reg_* b_cons
    rename b_* m2_*

    rangestat (reg) exret mktrf smb hml umd, interval(datemn -20 -5) by(permno)
    rename reg_nobs m3_nobs
    drop se_* reg_* b_cons
    rename b_* m3_*

    keep if m1_nobs >= 10 & !missing(m1_nobs)
    keep if datemn >= mofd(mdy(9,1,2018))
    drop *_nobs
    keep permno datemn m1* m2* m3*

    save testbetares4.dta, replace
restore

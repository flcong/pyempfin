clear all

use testfm.dta, clear
gen dateyr = substr(datemn, 1, 4)
gen datemnth = substr(datemn, 6, 7)
destring dateyr datemnth, replace
gen datemn2 = mofd(mdy(datemnth, 1, dateyr))
format datemn2 %tm
drop if missing(mktrf)
drop datemnth dateyr index datemn
rename datemn2 datemn

keep if datemn < 698
sort datemn permno
save asregtest.dta, replace

xtset permno datemn2

asreg exret mktrf smb hml, by(datemn2)

reg exret mktrf smb hml if datemn2 == 700

asreg exret mktrf smb hml, fmb save(firststage)

* Generate test data set for pyempfin;
* Select monthly returns of some stocks and the Fama-French three factors;

libname crsp "E:\MyCloud\Dropbox\Research\Data\wrds\crsp\20200623";
libname ff "E:\MyCloud\Dropbox\Research\Data\wrds\ff";

* Get CRSP data;
* Stocks: MSFT (10107), SVMK (18097), ZM (18484), JPM (47896), FDC (15703);
data msf; set crsp.msf;
	keep date permno ret;
	where permno in (10107, 18097, 18484, 47896, 15703) and 2017<=year(date)<=2019 and not missing(ret);
run;
* Merge with ticker;
proc sql;
	create table msf2 as
	select a.*, b.ticker
	from msf a left join crsp.stocknames b
	on a.permno=b.permno and b.namedt<=a.date<=b.nameenddt
	order by b.ticker, a.date;
quit;
* Calculate excess return;
proc sql;
	create table msf3 as
	select a.*, b.rf, a.ret-b.rf as exret
	from msf2 a left join ff.factors_monthly b
	on year(a.date)=year(b.date) and month(a.date)=month(b.date)
	order by a.ticker, a.date;
quit;

proc export data=msf3 outfile="E:\MyCloud\Dropbox\Research\Techniques\Github\pyempfin\pyempfin\tests\stktestdata.csv" dbms=csv replace;
run;

data ff; set ff.factors_monthly;
	keep date mktrf smb hml umd rf;
	where 2015<=year(date)<=2020;
run;

proc export data=ff outfile="E:\MyCloud\Dropbox\Research\Techniques\Github\pyempfin\pyempfin\tests\ff.csv" dbms=csv replace;
run;


* Test data for FM regression

* Get CRSP data;
data xsmsf; set crsp.msf;
	keep date permno ret;
	where 2017<=year(date)<=2019 and not missing(ret);
run;
* Keep S&P 500 stocks;
proc sql;
	create table xsmsf2 as
	select a.*
	from xsmsf a, crsp.msp500list b
	where a.permno=b.permno and b.start<=a.date<=b.ending
	order by a.permno, a.date;
quit;
* Calculate excess return;
proc sql;
	create table xsmsf3 as
	select a.*, b.rf, a.ret-b.rf as exret
	from xsmsf2 a left join ff.factors_monthly b
	on year(a.date)=year(b.date) and month(a.date)=month(b.date)
	order by a.permno, a.date;
quit;

proc export data=xsmsf3 outfile="E:\MyCloud\Dropbox\Research\Techniques\Github\pyempfin\pyempfin\tests\xsstktestdata.csv" dbms=csv replace;
run;

import pandas as pd
from typing import Union


def getffi48(sic: int, getdesc=False) -> Union[int,pd._libs.missing.NAType]:
    """Return Fama-French 48 industry code or description given SIC"""
    if pd.isna(sic):
        ffi48 = pd.NA
        desc = ''
    elif (100 <= sic <= 199) \
            or (200 <= sic <= 299) \
            or (700 <= sic <= 799) \
            or (910 <= sic <= 919) \
            or (2048 <= sic <= 2048):
        ffi48 = 1
        desc = 'Agriculture'
    elif (2000 <= sic <= 2009) \
            or (2010 <= sic <= 2019) \
            or (2020 <= sic <= 2029) \
            or (2030 <= sic <= 2039) \
            or (2040 <= sic <= 2046) \
            or (2050 <= sic <= 2059) \
            or (2060 <= sic <= 2063) \
            or (2070 <= sic <= 2079) \
            or (2090 <= sic <= 2092) \
            or (2095 <= sic <= 2095) \
            or (2098 <= sic <= 2099):
        ffi48 = 2
        desc = 'Food Products'
    elif (2064 <= sic <= 2068) \
            or (2086 <= sic <= 2086) \
            or (2087 <= sic <= 2087) \
            or (2096 <= sic <= 2096) \
            or (2097 <= sic <= 2097):
        ffi48 = 3
        desc = 'Candy & Soda'
    elif (2080 <= sic <= 2080) \
            or (2082 <= sic <= 2082) \
            or (2083 <= sic <= 2083) \
            or (2084 <= sic <= 2084) \
            or (2085 <= sic <= 2085):
        ffi48 = 4
        desc = 'Beer & Liquor'
    elif (2100 <= sic <= 2199):
        ffi48 = 5
        desc = 'Tobacco Products'
    elif (920 <= sic <= 999) \
            or (3650 <= sic <= 3651) \
            or (3652 <= sic <= 3652) \
            or (3732 <= sic <= 3732) \
            or (3930 <= sic <= 3931) \
            or (3940 <= sic <= 3949):
        ffi48 = 6
        desc = 'Recreation'
    elif (7800 <= sic <= 7829) \
            or (7830 <= sic <= 7833) \
            or (7840 <= sic <= 7841) \
            or (7900 <= sic <= 7900) \
            or (7910 <= sic <= 7911) \
            or (7920 <= sic <= 7929) \
            or (7930 <= sic <= 7933) \
            or (7940 <= sic <= 7949) \
            or (7980 <= sic <= 7980) \
            or (7990 <= sic <= 7999):
        ffi48 = 7
        desc = 'Entertainment'
    elif (2700 <= sic <= 2709) \
            or (2710 <= sic <= 2719) \
            or (2720 <= sic <= 2729) \
            or (2730 <= sic <= 2739) \
            or (2740 <= sic <= 2749) \
            or (2770 <= sic <= 2771) \
            or (2780 <= sic <= 2789) \
            or (2790 <= sic <= 2799):
        ffi48 = 8
        desc = 'Printing and Publishing'
    elif (2047 <= sic <= 2047) \
            or (2391 <= sic <= 2392) \
            or (2510 <= sic <= 2519) \
            or (2590 <= sic <= 2599) \
            or (2840 <= sic <= 2843) \
            or (2844 <= sic <= 2844) \
            or (3160 <= sic <= 3161) \
            or (3170 <= sic <= 3171) \
            or (3172 <= sic <= 3172) \
            or (3190 <= sic <= 3199) \
            or (3229 <= sic <= 3229) \
            or (3260 <= sic <= 3260) \
            or (3262 <= sic <= 3263) \
            or (3269 <= sic <= 3269) \
            or (3230 <= sic <= 3231) \
            or (3630 <= sic <= 3639) \
            or (3750 <= sic <= 3751) \
            or (3800 <= sic <= 3800) \
            or (3860 <= sic <= 3861) \
            or (3870 <= sic <= 3873) \
            or (3910 <= sic <= 3911) \
            or (3914 <= sic <= 3914) \
            or (3915 <= sic <= 3915) \
            or (3960 <= sic <= 3962) \
            or (3991 <= sic <= 3991) \
            or (3995 <= sic <= 3995):
        ffi48 = 9
        desc = 'Consumer Goods'
    elif (2300 <= sic <= 2390) \
            or (3020 <= sic <= 3021) \
            or (3100 <= sic <= 3111) \
            or (3130 <= sic <= 3131) \
            or (3140 <= sic <= 3149) \
            or (3150 <= sic <= 3151) \
            or (3963 <= sic <= 3965):
        ffi48 = 10
        desc = 'Apparel'
    elif (8000 <= sic <= 8099):
        ffi48 = 11
        desc = 'Healthcare'
    elif (3693 <= sic <= 3693) \
            or (3840 <= sic <= 3849) \
            or (3850 <= sic <= 3851):
        ffi48 = 12
        desc = 'Medical Equipment'
    elif (2830 <= sic <= 2830) \
            or (2831 <= sic <= 2831) \
            or (2833 <= sic <= 2833) \
            or (2834 <= sic <= 2834) \
            or (2835 <= sic <= 2835) \
            or (2836 <= sic <= 2836):
        ffi48 = 13
        desc = 'Pharmaceutical Products'
    elif (2800 <= sic <= 2809) \
            or (2810 <= sic <= 2819) \
            or (2820 <= sic <= 2829) \
            or (2850 <= sic <= 2859) \
            or (2860 <= sic <= 2869) \
            or (2870 <= sic <= 2879) \
            or (2890 <= sic <= 2899):
        ffi48 = 14
        desc = 'Chemicals'
    elif (3031 <= sic <= 3031) \
            or (3041 <= sic <= 3041) \
            or (3050 <= sic <= 3053) \
            or (3060 <= sic <= 3069) \
            or (3070 <= sic <= 3079) \
            or (3080 <= sic <= 3089) \
            or (3090 <= sic <= 3099):
        ffi48 = 15
        desc = 'Rubber and Plastic Products'
    elif (2200 <= sic <= 2269) \
            or (2270 <= sic <= 2279) \
            or (2280 <= sic <= 2284) \
            or (2290 <= sic <= 2295) \
            or (2297 <= sic <= 2297) \
            or (2298 <= sic <= 2298) \
            or (2299 <= sic <= 2299) \
            or (2393 <= sic <= 2395) \
            or (2397 <= sic <= 2399):
        ffi48 = 16
        desc = 'Textiles'
    elif (800 <= sic <= 899) \
            or (2400 <= sic <= 2439) \
            or (2450 <= sic <= 2459) \
            or (2490 <= sic <= 2499) \
            or (2660 <= sic <= 2661) \
            or (2950 <= sic <= 2952) \
            or (3200 <= sic <= 3200) \
            or (3210 <= sic <= 3211) \
            or (3240 <= sic <= 3241) \
            or (3250 <= sic <= 3259) \
            or (3261 <= sic <= 3261) \
            or (3264 <= sic <= 3264) \
            or (3270 <= sic <= 3275) \
            or (3280 <= sic <= 3281) \
            or (3290 <= sic <= 3293) \
            or (3295 <= sic <= 3299) \
            or (3420 <= sic <= 3429) \
            or (3430 <= sic <= 3433) \
            or (3440 <= sic <= 3441) \
            or (3442 <= sic <= 3442) \
            or (3446 <= sic <= 3446) \
            or (3448 <= sic <= 3448) \
            or (3449 <= sic <= 3449) \
            or (3450 <= sic <= 3451) \
            or (3452 <= sic <= 3452) \
            or (3490 <= sic <= 3499) \
            or (3996 <= sic <= 3996):
        ffi48 = 17
        desc = 'Construction Materials'
    elif (1500 <= sic <= 1511) \
            or (1520 <= sic <= 1529) \
            or (1530 <= sic <= 1539) \
            or (1540 <= sic <= 1549) \
            or (1600 <= sic <= 1699) \
            or (1700 <= sic <= 1799):
        ffi48 = 18
        desc = 'Construction'
    elif (3300 <= sic <= 3300) \
            or (3310 <= sic <= 3317) \
            or (3320 <= sic <= 3325) \
            or (3330 <= sic <= 3339) \
            or (3340 <= sic <= 3341) \
            or (3350 <= sic <= 3357) \
            or (3360 <= sic <= 3369) \
            or (3370 <= sic <= 3379) \
            or (3390 <= sic <= 3399):
        ffi48 = 19
        desc = 'Steel Works Etc'
    elif (3400 <= sic <= 3400) \
            or (3443 <= sic <= 3443) \
            or (3444 <= sic <= 3444) \
            or (3460 <= sic <= 3469) \
            or (3470 <= sic <= 3479):
        ffi48 = 20
        desc = 'Fabricated Products'
    elif (3510 <= sic <= 3519) \
            or (3520 <= sic <= 3529) \
            or (3530 <= sic <= 3530) \
            or (3531 <= sic <= 3531) \
            or (3532 <= sic <= 3532) \
            or (3533 <= sic <= 3533) \
            or (3534 <= sic <= 3534) \
            or (3535 <= sic <= 3535) \
            or (3536 <= sic <= 3536) \
            or (3538 <= sic <= 3538) \
            or (3540 <= sic <= 3549) \
            or (3550 <= sic <= 3559) \
            or (3560 <= sic <= 3569) \
            or (3580 <= sic <= 3580) \
            or (3581 <= sic <= 3581) \
            or (3582 <= sic <= 3582) \
            or (3585 <= sic <= 3585) \
            or (3586 <= sic <= 3586) \
            or (3589 <= sic <= 3589) \
            or (3590 <= sic <= 3599):
        ffi48 = 21
        desc = 'Machinery'
    elif (3600 <= sic <= 3600) \
            or (3610 <= sic <= 3613) \
            or (3620 <= sic <= 3621) \
            or (3623 <= sic <= 3629) \
            or (3640 <= sic <= 3644) \
            or (3645 <= sic <= 3645) \
            or (3646 <= sic <= 3646) \
            or (3648 <= sic <= 3649) \
            or (3660 <= sic <= 3660) \
            or (3690 <= sic <= 3690) \
            or (3691 <= sic <= 3692) \
            or (3699 <= sic <= 3699):
        ffi48 = 22
        desc = 'Electrical Equipment'
    elif (2296 <= sic <= 2296) \
            or (2396 <= sic <= 2396) \
            or (3010 <= sic <= 3011) \
            or (3537 <= sic <= 3537) \
            or (3647 <= sic <= 3647) \
            or (3694 <= sic <= 3694) \
            or (3700 <= sic <= 3700) \
            or (3710 <= sic <= 3710) \
            or (3711 <= sic <= 3711) \
            or (3713 <= sic <= 3713) \
            or (3714 <= sic <= 3714) \
            or (3715 <= sic <= 3715) \
            or (3716 <= sic <= 3716) \
            or (3792 <= sic <= 3792) \
            or (3790 <= sic <= 3791) \
            or (3799 <= sic <= 3799):
        ffi48 = 23
        desc = 'Automobiles and Trucks'
    elif (3720 <= sic <= 3720) \
            or (3721 <= sic <= 3721) \
            or (3723 <= sic <= 3724) \
            or (3725 <= sic <= 3725) \
            or (3728 <= sic <= 3729):
        ffi48 = 24
        desc = 'Aircraft'
    elif (3730 <= sic <= 3731) \
            or (3740 <= sic <= 3743):
        ffi48 = 25
        desc = 'Shipbuilding, Railroad Equipment'
    elif (3760 <= sic <= 3769) \
            or (3795 <= sic <= 3795) \
            or (3480 <= sic <= 3489):
        ffi48 = 26
        desc = 'Defense'
    elif (1040 <= sic <= 1049):
        ffi48 = 27
        desc = 'Precious Metals'
    elif (1000 <= sic <= 1009) \
            or (1010 <= sic <= 1019) \
            or (1020 <= sic <= 1029) \
            or (1030 <= sic <= 1039) \
            or (1050 <= sic <= 1059) \
            or (1060 <= sic <= 1069) \
            or (1070 <= sic <= 1079) \
            or (1080 <= sic <= 1089) \
            or (1090 <= sic <= 1099) \
            or (1100 <= sic <= 1119) \
            or (1400 <= sic <= 1499):
        ffi48 = 28
        desc = 'Non-Metallic and Industrial Metal Mining'
    elif (1200 <= sic <= 1299):
        ffi48 = 29
        desc = 'Coal'
    elif (1300 <= sic <= 1300) \
            or (1310 <= sic <= 1319) \
            or (1320 <= sic <= 1329) \
            or (1330 <= sic <= 1339) \
            or (1370 <= sic <= 1379) \
            or (1380 <= sic <= 1380) \
            or (1381 <= sic <= 1381) \
            or (1382 <= sic <= 1382) \
            or (1389 <= sic <= 1389) \
            or (2900 <= sic <= 2912) \
            or (2990 <= sic <= 2999):
        ffi48 = 30
        desc = 'Petroleum and Natural Gas'
    elif (4900 <= sic <= 4900) \
            or (4910 <= sic <= 4911) \
            or (4920 <= sic <= 4922) \
            or (4923 <= sic <= 4923) \
            or (4924 <= sic <= 4925) \
            or (4930 <= sic <= 4931) \
            or (4932 <= sic <= 4932) \
            or (4939 <= sic <= 4939) \
            or (4940 <= sic <= 4942):
        ffi48 = 31
        desc = 'Utilities'
    elif (4800 <= sic <= 4800) \
            or (4810 <= sic <= 4813) \
            or (4820 <= sic <= 4822) \
            or (4830 <= sic <= 4839) \
            or (4840 <= sic <= 4841) \
            or (4880 <= sic <= 4889) \
            or (4890 <= sic <= 4890) \
            or (4891 <= sic <= 4891) \
            or (4892 <= sic <= 4892) \
            or (4899 <= sic <= 4899):
        ffi48 = 32
        desc = 'Communication'
    elif (7020 <= sic <= 7021) \
            or (7030 <= sic <= 7033) \
            or (7200 <= sic <= 7200) \
            or (7210 <= sic <= 7212) \
            or (7214 <= sic <= 7214) \
            or (7215 <= sic <= 7216) \
            or (7217 <= sic <= 7217) \
            or (7219 <= sic <= 7219) \
            or (7220 <= sic <= 7221) \
            or (7230 <= sic <= 7231) \
            or (7240 <= sic <= 7241) \
            or (7250 <= sic <= 7251) \
            or (7260 <= sic <= 7269) \
            or (7270 <= sic <= 7290) \
            or (7291 <= sic <= 7291) \
            or (7292 <= sic <= 7299) \
            or (7395 <= sic <= 7395) \
            or (7500 <= sic <= 7500) \
            or (7520 <= sic <= 7529) \
            or (7530 <= sic <= 7539) \
            or (7540 <= sic <= 7549) \
            or (7600 <= sic <= 7600) \
            or (7620 <= sic <= 7620) \
            or (7622 <= sic <= 7622) \
            or (7623 <= sic <= 7623) \
            or (7629 <= sic <= 7629) \
            or (7630 <= sic <= 7631) \
            or (7640 <= sic <= 7641) \
            or (7690 <= sic <= 7699) \
            or (8100 <= sic <= 8199) \
            or (8200 <= sic <= 8299) \
            or (8300 <= sic <= 8399) \
            or (8400 <= sic <= 8499) \
            or (8600 <= sic <= 8699) \
            or (8800 <= sic <= 8899) \
            or (7510 <= sic <= 7515):
        ffi48 = 33
        desc = 'Personal Services'
    elif (2750 <= sic <= 2759) \
            or (3993 <= sic <= 3993) \
            or (7218 <= sic <= 7218) \
            or (7300 <= sic <= 7300) \
            or (7310 <= sic <= 7319) \
            or (7320 <= sic <= 7329) \
            or (7330 <= sic <= 7339) \
            or (7340 <= sic <= 7342) \
            or (7349 <= sic <= 7349) \
            or (7350 <= sic <= 7351) \
            or (7352 <= sic <= 7352) \
            or (7353 <= sic <= 7353) \
            or (7359 <= sic <= 7359) \
            or (7360 <= sic <= 7369) \
            or (7370 <= sic <= 7372) \
            or (7374 <= sic <= 7374) \
            or (7375 <= sic <= 7375) \
            or (7376 <= sic <= 7376) \
            or (7377 <= sic <= 7377) \
            or (7378 <= sic <= 7378) \
            or (7379 <= sic <= 7379) \
            or (7380 <= sic <= 7380) \
            or (7381 <= sic <= 7382) \
            or (7383 <= sic <= 7383) \
            or (7384 <= sic <= 7384) \
            or (7385 <= sic <= 7385) \
            or (7389 <= sic <= 7390) \
            or (7391 <= sic <= 7391) \
            or (7392 <= sic <= 7392) \
            or (7393 <= sic <= 7393) \
            or (7394 <= sic <= 7394) \
            or (7396 <= sic <= 7396) \
            or (7397 <= sic <= 7397) \
            or (7399 <= sic <= 7399) \
            or (7519 <= sic <= 7519) \
            or (8700 <= sic <= 8700) \
            or (8710 <= sic <= 8713) \
            or (8720 <= sic <= 8721) \
            or (8730 <= sic <= 8734) \
            or (8740 <= sic <= 8748) \
            or (8900 <= sic <= 8910) \
            or (8911 <= sic <= 8911) \
            or (8920 <= sic <= 8999) \
            or (4220 <= sic <= 4229):
        ffi48 = 34
        desc = 'Business Services'
    elif (3570 <= sic <= 3579) \
            or (3680 <= sic <= 3680) \
            or (3681 <= sic <= 3681) \
            or (3682 <= sic <= 3682) \
            or (3683 <= sic <= 3683) \
            or (3684 <= sic <= 3684) \
            or (3685 <= sic <= 3685) \
            or (3686 <= sic <= 3686) \
            or (3687 <= sic <= 3687) \
            or (3688 <= sic <= 3688) \
            or (3689 <= sic <= 3689) \
            or (3695 <= sic <= 3695) \
            or (7373 <= sic <= 7373):
        ffi48 = 35
        desc = 'Computers'
    elif (3622 <= sic <= 3622) \
            or (3661 <= sic <= 3661) \
            or (3662 <= sic <= 3662) \
            or (3663 <= sic <= 3663) \
            or (3664 <= sic <= 3664) \
            or (3665 <= sic <= 3665) \
            or (3666 <= sic <= 3666) \
            or (3669 <= sic <= 3669) \
            or (3670 <= sic <= 3679) \
            or (3810 <= sic <= 3810) \
            or (3812 <= sic <= 3812):
        ffi48 = 36
        desc = 'Electronic Equipment'
    elif (3811 <= sic <= 3811) \
            or (3820 <= sic <= 3820) \
            or (3821 <= sic <= 3821) \
            or (3822 <= sic <= 3822) \
            or (3823 <= sic <= 3823) \
            or (3824 <= sic <= 3824) \
            or (3825 <= sic <= 3825) \
            or (3826 <= sic <= 3826) \
            or (3827 <= sic <= 3827) \
            or (3829 <= sic <= 3829) \
            or (3830 <= sic <= 3839):
        ffi48 = 37
        desc = 'Measuring and Control Equipment'
    elif (2520 <= sic <= 2549) \
            or (2600 <= sic <= 2639) \
            or (2670 <= sic <= 2699) \
            or (2760 <= sic <= 2761) \
            or (3950 <= sic <= 3955):
        ffi48 = 38
        desc = 'Business Supplies'
    elif (2440 <= sic <= 2449) \
            or (2640 <= sic <= 2659) \
            or (3220 <= sic <= 3221) \
            or (3410 <= sic <= 3412):
        ffi48 = 39
        desc = 'Shipping Containers'
    elif (4000 <= sic <= 4013) \
            or (4040 <= sic <= 4049) \
            or (4100 <= sic <= 4100) \
            or (4110 <= sic <= 4119) \
            or (4120 <= sic <= 4121) \
            or (4130 <= sic <= 4131) \
            or (4140 <= sic <= 4142) \
            or (4150 <= sic <= 4151) \
            or (4170 <= sic <= 4173) \
            or (4190 <= sic <= 4199) \
            or (4200 <= sic <= 4200) \
            or (4210 <= sic <= 4219) \
            or (4230 <= sic <= 4231) \
            or (4240 <= sic <= 4249) \
            or (4400 <= sic <= 4499) \
            or (4500 <= sic <= 4599) \
            or (4600 <= sic <= 4699) \
            or (4700 <= sic <= 4700) \
            or (4710 <= sic <= 4712) \
            or (4720 <= sic <= 4729) \
            or (4730 <= sic <= 4739) \
            or (4740 <= sic <= 4749) \
            or (4780 <= sic <= 4780) \
            or (4782 <= sic <= 4782) \
            or (4783 <= sic <= 4783) \
            or (4784 <= sic <= 4784) \
            or (4785 <= sic <= 4785) \
            or (4789 <= sic <= 4789):
        ffi48 = 40
        desc = 'Transportation'
    elif (5000 <= sic <= 5000) \
            or (5010 <= sic <= 5015) \
            or (5020 <= sic <= 5023) \
            or (5030 <= sic <= 5039) \
            or (5040 <= sic <= 5042) \
            or (5043 <= sic <= 5043) \
            or (5044 <= sic <= 5044) \
            or (5045 <= sic <= 5045) \
            or (5046 <= sic <= 5046) \
            or (5047 <= sic <= 5047) \
            or (5048 <= sic <= 5048) \
            or (5049 <= sic <= 5049) \
            or (5050 <= sic <= 5059) \
            or (5060 <= sic <= 5060) \
            or (5063 <= sic <= 5063) \
            or (5064 <= sic <= 5064) \
            or (5065 <= sic <= 5065) \
            or (5070 <= sic <= 5078) \
            or (5080 <= sic <= 5080) \
            or (5081 <= sic <= 5081) \
            or (5082 <= sic <= 5082) \
            or (5083 <= sic <= 5083) \
            or (5084 <= sic <= 5084) \
            or (5085 <= sic <= 5085) \
            or (5086 <= sic <= 5087) \
            or (5088 <= sic <= 5088) \
            or (5090 <= sic <= 5090) \
            or (5091 <= sic <= 5092) \
            or (5093 <= sic <= 5093) \
            or (5094 <= sic <= 5094) \
            or (5099 <= sic <= 5099) \
            or (5100 <= sic <= 5100) \
            or (5110 <= sic <= 5113) \
            or (5120 <= sic <= 5122) \
            or (5130 <= sic <= 5139) \
            or (5140 <= sic <= 5149) \
            or (5150 <= sic <= 5159) \
            or (5160 <= sic <= 5169) \
            or (5170 <= sic <= 5172) \
            or (5180 <= sic <= 5182) \
            or (5190 <= sic <= 5199):
        ffi48 = 41
        desc = 'Wholesale'
    elif (5200 <= sic <= 5200) \
            or (5210 <= sic <= 5219) \
            or (5220 <= sic <= 5229) \
            or (5230 <= sic <= 5231) \
            or (5250 <= sic <= 5251) \
            or (5260 <= sic <= 5261) \
            or (5270 <= sic <= 5271) \
            or (5300 <= sic <= 5300) \
            or (5310 <= sic <= 5311) \
            or (5320 <= sic <= 5320) \
            or (5330 <= sic <= 5331) \
            or (5334 <= sic <= 5334) \
            or (5340 <= sic <= 5349) \
            or (5390 <= sic <= 5399) \
            or (5400 <= sic <= 5400) \
            or (5410 <= sic <= 5411) \
            or (5412 <= sic <= 5412) \
            or (5420 <= sic <= 5429) \
            or (5430 <= sic <= 5439) \
            or (5440 <= sic <= 5449) \
            or (5450 <= sic <= 5459) \
            or (5460 <= sic <= 5469) \
            or (5490 <= sic <= 5499) \
            or (5500 <= sic <= 5500) \
            or (5510 <= sic <= 5529) \
            or (5530 <= sic <= 5539) \
            or (5540 <= sic <= 5549) \
            or (5550 <= sic <= 5559) \
            or (5560 <= sic <= 5569) \
            or (5570 <= sic <= 5579) \
            or (5590 <= sic <= 5599) \
            or (5600 <= sic <= 5699) \
            or (5700 <= sic <= 5700) \
            or (5710 <= sic <= 5719) \
            or (5720 <= sic <= 5722) \
            or (5730 <= sic <= 5733) \
            or (5734 <= sic <= 5734) \
            or (5735 <= sic <= 5735) \
            or (5736 <= sic <= 5736) \
            or (5750 <= sic <= 5799) \
            or (5900 <= sic <= 5900) \
            or (5910 <= sic <= 5912) \
            or (5920 <= sic <= 5929) \
            or (5930 <= sic <= 5932) \
            or (5940 <= sic <= 5940) \
            or (5941 <= sic <= 5941) \
            or (5942 <= sic <= 5942) \
            or (5943 <= sic <= 5943) \
            or (5944 <= sic <= 5944) \
            or (5945 <= sic <= 5945) \
            or (5946 <= sic <= 5946) \
            or (5947 <= sic <= 5947) \
            or (5948 <= sic <= 5948) \
            or (5949 <= sic <= 5949) \
            or (5950 <= sic <= 5959) \
            or (5960 <= sic <= 5969) \
            or (5970 <= sic <= 5979) \
            or (5980 <= sic <= 5989) \
            or (5990 <= sic <= 5990) \
            or (5992 <= sic <= 5992) \
            or (5993 <= sic <= 5993) \
            or (5994 <= sic <= 5994) \
            or (5995 <= sic <= 5995) \
            or (5999 <= sic <= 5999):
        ffi48 = 42
        desc = 'Retail'
    elif (5800 <= sic <= 5819) \
            or (5820 <= sic <= 5829) \
            or (5890 <= sic <= 5899) \
            or (7000 <= sic <= 7000) \
            or (7010 <= sic <= 7019) \
            or (7040 <= sic <= 7049) \
            or (7213 <= sic <= 7213):
        ffi48 = 43
        desc = 'Restaurants, Hotels, Motels'
    elif (6000 <= sic <= 6000) \
            or (6010 <= sic <= 6019) \
            or (6020 <= sic <= 6020) \
            or (6021 <= sic <= 6021) \
            or (6022 <= sic <= 6022) \
            or (6023 <= sic <= 6024) \
            or (6025 <= sic <= 6025) \
            or (6026 <= sic <= 6026) \
            or (6027 <= sic <= 6027) \
            or (6028 <= sic <= 6029) \
            or (6030 <= sic <= 6036) \
            or (6040 <= sic <= 6059) \
            or (6060 <= sic <= 6062) \
            or (6080 <= sic <= 6082) \
            or (6090 <= sic <= 6099) \
            or (6100 <= sic <= 6100) \
            or (6110 <= sic <= 6111) \
            or (6112 <= sic <= 6113) \
            or (6120 <= sic <= 6129) \
            or (6130 <= sic <= 6139) \
            or (6140 <= sic <= 6149) \
            or (6150 <= sic <= 6159) \
            or (6160 <= sic <= 6169) \
            or (6170 <= sic <= 6179) \
            or (6190 <= sic <= 6199):
        ffi48 = 44
        desc = 'Banking'
    elif (6300 <= sic <= 6300) \
            or (6310 <= sic <= 6319) \
            or (6320 <= sic <= 6329) \
            or (6330 <= sic <= 6331) \
            or (6350 <= sic <= 6351) \
            or (6360 <= sic <= 6361) \
            or (6370 <= sic <= 6379) \
            or (6390 <= sic <= 6399) \
            or (6400 <= sic <= 6411):
        ffi48 = 45
        desc = 'Insurance'
    elif (6500 <= sic <= 6500) \
            or (6510 <= sic <= 6510) \
            or (6512 <= sic <= 6512) \
            or (6513 <= sic <= 6513) \
            or (6514 <= sic <= 6514) \
            or (6515 <= sic <= 6515) \
            or (6517 <= sic <= 6519) \
            or (6520 <= sic <= 6529) \
            or (6530 <= sic <= 6531) \
            or (6532 <= sic <= 6532) \
            or (6540 <= sic <= 6541) \
            or (6550 <= sic <= 6553) \
            or (6590 <= sic <= 6599) \
            or (6610 <= sic <= 6611):
        ffi48 = 46
        desc = 'Real Estate'
    elif (6200 <= sic <= 6299) \
            or (6700 <= sic <= 6700) \
            or (6710 <= sic <= 6719) \
            or (6720 <= sic <= 6722) \
            or (6723 <= sic <= 6723) \
            or (6724 <= sic <= 6724) \
            or (6725 <= sic <= 6725) \
            or (6726 <= sic <= 6726) \
            or (6730 <= sic <= 6733) \
            or (6740 <= sic <= 6779) \
            or (6790 <= sic <= 6791) \
            or (6792 <= sic <= 6792) \
            or (6793 <= sic <= 6793) \
            or (6794 <= sic <= 6794) \
            or (6795 <= sic <= 6795) \
            or (6798 <= sic <= 6798) \
            or (6799 <= sic <= 6799):
        ffi48 = 47
        desc = 'Trading'
    elif (4950 <= sic <= 4959) \
            or (4960 <= sic <= 4961) \
            or (4970 <= sic <= 4971) \
            or (4990 <= sic <= 4991):
        ffi48 = 48
        desc = 'Almost Nothing'
    else:
        ffi48 = pd.NA
        desc = ''
    if getdesc:
        return desc
    else:
        return ffi48


def getffi30(sic: int, getdesc=False) -> Union[int,pd._libs.missing.NAType]:
    """Return Fama-French 30 industry code or description given SIC"""
    if pd.isna(sic):
        ffi30 = pd.NA
        desc = ''
    elif (100 <= sic <= 199) \
            or (200 <= sic <= 299) \
            or (700 <= sic <= 799) \
            or (910 <= sic <= 919) \
            or (2000 <= sic <= 2009) \
            or (2010 <= sic <= 2019) \
            or (2020 <= sic <= 2029) \
            or (2030 <= sic <= 2039) \
            or (2040 <= sic <= 2046) \
            or (2048 <= sic <= 2048) \
            or (2050 <= sic <= 2059) \
            or (2060 <= sic <= 2063) \
            or (2064 <= sic <= 2068) \
            or (2070 <= sic <= 2079) \
            or (2086 <= sic <= 2086) \
            or (2087 <= sic <= 2087) \
            or (2090 <= sic <= 2092) \
            or (2095 <= sic <= 2095) \
            or (2096 <= sic <= 2096) \
            or (2097 <= sic <= 2097) \
            or (2098 <= sic <= 2099):
        ffi30 = 1
        desc = 'Food Products'
    elif (2080 <= sic <= 2080) \
            or (2082 <= sic <= 2082) \
            or (2083 <= sic <= 2083) \
            or (2084 <= sic <= 2084) \
            or (2085 <= sic <= 2085):
        ffi30 = 2
        desc = 'Beer & Liquor'
    elif (2100 <= sic <= 2199):
        ffi30 = 3
        desc = 'Tobacco Products'
    elif (920 <= sic <= 999) \
            or (3650 <= sic <= 3651) \
            or (3652 <= sic <= 3652) \
            or (3732 <= sic <= 3732) \
            or (3930 <= sic <= 3931) \
            or (3940 <= sic <= 3949) \
            or (7800 <= sic <= 7829) \
            or (7830 <= sic <= 7833) \
            or (7840 <= sic <= 7841) \
            or (7900 <= sic <= 7900) \
            or (7910 <= sic <= 7911) \
            or (7920 <= sic <= 7929) \
            or (7930 <= sic <= 7933) \
            or (7940 <= sic <= 7949) \
            or (7980 <= sic <= 7980) \
            or (7990 <= sic <= 7999):
        ffi30 = 4
        desc = 'Recreation'
    elif (2700 <= sic <= 2709) \
            or (2710 <= sic <= 2719) \
            or (2720 <= sic <= 2729) \
            or (2730 <= sic <= 2739) \
            or (2740 <= sic <= 2749) \
            or (2750 <= sic <= 2759) \
            or (2770 <= sic <= 2771) \
            or (2780 <= sic <= 2789) \
            or (2790 <= sic <= 2799) \
            or (3993 <= sic <= 3993):
        ffi30 = 5
        desc = 'Printing and Publishing'
    elif (2047 <= sic <= 2047) \
            or (2391 <= sic <= 2392) \
            or (2510 <= sic <= 2519) \
            or (2590 <= sic <= 2599) \
            or (2840 <= sic <= 2843) \
            or (2844 <= sic <= 2844) \
            or (3160 <= sic <= 3161) \
            or (3170 <= sic <= 3171) \
            or (3172 <= sic <= 3172) \
            or (3190 <= sic <= 3199) \
            or (3229 <= sic <= 3229) \
            or (3260 <= sic <= 3260) \
            or (3262 <= sic <= 3263) \
            or (3269 <= sic <= 3269) \
            or (3230 <= sic <= 3231) \
            or (3630 <= sic <= 3639) \
            or (3750 <= sic <= 3751) \
            or (3800 <= sic <= 3800) \
            or (3860 <= sic <= 3861) \
            or (3870 <= sic <= 3873) \
            or (3910 <= sic <= 3911) \
            or (3914 <= sic <= 3914) \
            or (3915 <= sic <= 3915) \
            or (3960 <= sic <= 3962) \
            or (3991 <= sic <= 3991) \
            or (3995 <= sic <= 3995):
        ffi30 = 6
        desc = 'Consumer Goods'
    elif (2300 <= sic <= 2390) \
            or (3020 <= sic <= 3021) \
            or (3100 <= sic <= 3111) \
            or (3130 <= sic <= 3131) \
            or (3140 <= sic <= 3149) \
            or (3150 <= sic <= 3151) \
            or (3963 <= sic <= 3965):
        ffi30 = 7
        desc = 'Apparel'
    elif (2830 <= sic <= 2830) \
            or (2831 <= sic <= 2831) \
            or (2833 <= sic <= 2833) \
            or (2834 <= sic <= 2834) \
            or (2835 <= sic <= 2835) \
            or (2836 <= sic <= 2836) \
            or (3693 <= sic <= 3693) \
            or (3840 <= sic <= 3849) \
            or (3850 <= sic <= 3851) \
            or (8000 <= sic <= 8099):
        ffi30 = 8
        desc = 'Healthcare, Medical Equipment, Pharmaceutical Products'
    elif (2800 <= sic <= 2809) \
            or (2810 <= sic <= 2819) \
            or (2820 <= sic <= 2829) \
            or (2850 <= sic <= 2859) \
            or (2860 <= sic <= 2869) \
            or (2870 <= sic <= 2879) \
            or (2890 <= sic <= 2899):
        ffi30 = 9
        desc = 'Chemicals'
    elif (2200 <= sic <= 2269) \
            or (2270 <= sic <= 2279) \
            or (2280 <= sic <= 2284) \
            or (2290 <= sic <= 2295) \
            or (2297 <= sic <= 2297) \
            or (2298 <= sic <= 2298) \
            or (2299 <= sic <= 2299) \
            or (2393 <= sic <= 2395) \
            or (2397 <= sic <= 2399):
        ffi30 = 10
        desc = 'Textiles'
    elif (800 <= sic <= 899) \
            or (1500 <= sic <= 1511) \
            or (1520 <= sic <= 1529) \
            or (1530 <= sic <= 1539) \
            or (1540 <= sic <= 1549) \
            or (1600 <= sic <= 1699) \
            or (1700 <= sic <= 1799) \
            or (2400 <= sic <= 2439) \
            or (2450 <= sic <= 2459) \
            or (2490 <= sic <= 2499) \
            or (2660 <= sic <= 2661) \
            or (2950 <= sic <= 2952) \
            or (3200 <= sic <= 3200) \
            or (3210 <= sic <= 3211) \
            or (3240 <= sic <= 3241) \
            or (3250 <= sic <= 3259) \
            or (3261 <= sic <= 3261) \
            or (3264 <= sic <= 3264) \
            or (3270 <= sic <= 3275) \
            or (3280 <= sic <= 3281) \
            or (3290 <= sic <= 3293) \
            or (3295 <= sic <= 3299) \
            or (3420 <= sic <= 3429) \
            or (3430 <= sic <= 3433) \
            or (3440 <= sic <= 3441) \
            or (3442 <= sic <= 3442) \
            or (3446 <= sic <= 3446) \
            or (3448 <= sic <= 3448) \
            or (3449 <= sic <= 3449) \
            or (3450 <= sic <= 3451) \
            or (3452 <= sic <= 3452) \
            or (3490 <= sic <= 3499) \
            or (3996 <= sic <= 3996):
        ffi30 = 11
        desc = 'Construction and Construction Materials'
    elif (3300 <= sic <= 3300) \
            or (3310 <= sic <= 3317) \
            or (3320 <= sic <= 3325) \
            or (3330 <= sic <= 3339) \
            or (3340 <= sic <= 3341) \
            or (3350 <= sic <= 3357) \
            or (3360 <= sic <= 3369) \
            or (3370 <= sic <= 3379) \
            or (3390 <= sic <= 3399):
        ffi30 = 12
        desc = 'Steel Works Etc'
    elif (3400 <= sic <= 3400) \
            or (3443 <= sic <= 3443) \
            or (3444 <= sic <= 3444) \
            or (3460 <= sic <= 3469) \
            or (3470 <= sic <= 3479) \
            or (3510 <= sic <= 3519) \
            or (3520 <= sic <= 3529) \
            or (3530 <= sic <= 3530) \
            or (3531 <= sic <= 3531) \
            or (3532 <= sic <= 3532) \
            or (3533 <= sic <= 3533) \
            or (3534 <= sic <= 3534) \
            or (3535 <= sic <= 3535) \
            or (3536 <= sic <= 3536) \
            or (3538 <= sic <= 3538) \
            or (3540 <= sic <= 3549) \
            or (3550 <= sic <= 3559) \
            or (3560 <= sic <= 3569) \
            or (3580 <= sic <= 3580) \
            or (3581 <= sic <= 3581) \
            or (3582 <= sic <= 3582) \
            or (3585 <= sic <= 3585) \
            or (3586 <= sic <= 3586) \
            or (3589 <= sic <= 3589) \
            or (3590 <= sic <= 3599):
        ffi30 = 13
        desc = 'Fabricated Products and Machinery'
    elif (3600 <= sic <= 3600) \
            or (3610 <= sic <= 3613) \
            or (3620 <= sic <= 3621) \
            or (3623 <= sic <= 3629) \
            or (3640 <= sic <= 3644) \
            or (3645 <= sic <= 3645) \
            or (3646 <= sic <= 3646) \
            or (3648 <= sic <= 3649) \
            or (3660 <= sic <= 3660) \
            or (3690 <= sic <= 3690) \
            or (3691 <= sic <= 3692) \
            or (3699 <= sic <= 3699):
        ffi30 = 14
        desc = 'Electrical Equipment'
    elif (2296 <= sic <= 2296) \
            or (2396 <= sic <= 2396) \
            or (3010 <= sic <= 3011) \
            or (3537 <= sic <= 3537) \
            or (3647 <= sic <= 3647) \
            or (3694 <= sic <= 3694) \
            or (3700 <= sic <= 3700) \
            or (3710 <= sic <= 3710) \
            or (3711 <= sic <= 3711) \
            or (3713 <= sic <= 3713) \
            or (3714 <= sic <= 3714) \
            or (3715 <= sic <= 3715) \
            or (3716 <= sic <= 3716) \
            or (3792 <= sic <= 3792) \
            or (3790 <= sic <= 3791) \
            or (3799 <= sic <= 3799):
        ffi30 = 15
        desc = 'Automobiles and Trucks'
    elif (3720 <= sic <= 3720) \
            or (3721 <= sic <= 3721) \
            or (3723 <= sic <= 3724) \
            or (3725 <= sic <= 3725) \
            or (3728 <= sic <= 3729) \
            or (3730 <= sic <= 3731) \
            or (3740 <= sic <= 3743):
        ffi30 = 16
        desc = 'Aircraft, ships, and railroad equipment'
    elif (1000 <= sic <= 1009) \
            or (1010 <= sic <= 1019) \
            or (1020 <= sic <= 1029) \
            or (1030 <= sic <= 1039) \
            or (1040 <= sic <= 1049) \
            or (1050 <= sic <= 1059) \
            or (1060 <= sic <= 1069) \
            or (1070 <= sic <= 1079) \
            or (1080 <= sic <= 1089) \
            or (1090 <= sic <= 1099) \
            or (1100 <= sic <= 1119) \
            or (1400 <= sic <= 1499):
        ffi30 = 17
        desc = 'Precious Metals, Non-Metallic, and Industrial Metal Mining'
    elif (1200 <= sic <= 1299):
        ffi30 = 18
        desc = 'Coal'
    elif (1300 <= sic <= 1300) \
            or (1310 <= sic <= 1319) \
            or (1320 <= sic <= 1329) \
            or (1330 <= sic <= 1339) \
            or (1370 <= sic <= 1379) \
            or (1380 <= sic <= 1380) \
            or (1381 <= sic <= 1381) \
            or (1382 <= sic <= 1382) \
            or (1389 <= sic <= 1389) \
            or (2900 <= sic <= 2912) \
            or (2990 <= sic <= 2999):
        ffi30 = 19
        desc = 'Petroleum and Natural Gas'
    elif (4900 <= sic <= 4900) \
            or (4910 <= sic <= 4911) \
            or (4920 <= sic <= 4922) \
            or (4923 <= sic <= 4923) \
            or (4924 <= sic <= 4925) \
            or (4930 <= sic <= 4931) \
            or (4932 <= sic <= 4932) \
            or (4939 <= sic <= 4939) \
            or (4940 <= sic <= 4942):
        ffi30 = 20
        desc = 'Utilities'
    elif (4800 <= sic <= 4800) \
            or (4810 <= sic <= 4813) \
            or (4820 <= sic <= 4822) \
            or (4830 <= sic <= 4839) \
            or (4840 <= sic <= 4841) \
            or (4880 <= sic <= 4889) \
            or (4890 <= sic <= 4890) \
            or (4891 <= sic <= 4891) \
            or (4892 <= sic <= 4892) \
            or (4899 <= sic <= 4899):
        ffi30 = 21
        desc = 'Communication'
    elif (7020 <= sic <= 7021) \
            or (7030 <= sic <= 7033) \
            or (7200 <= sic <= 7200) \
            or (7210 <= sic <= 7212) \
            or (7214 <= sic <= 7214) \
            or (7215 <= sic <= 7216) \
            or (7217 <= sic <= 7217) \
            or (7218 <= sic <= 7218) \
            or (7219 <= sic <= 7219) \
            or (7220 <= sic <= 7221) \
            or (7230 <= sic <= 7231) \
            or (7240 <= sic <= 7241) \
            or (7250 <= sic <= 7251) \
            or (7260 <= sic <= 7269) \
            or (7270 <= sic <= 7290) \
            or (7291 <= sic <= 7291) \
            or (7292 <= sic <= 7299) \
            or (7300 <= sic <= 7300) \
            or (7310 <= sic <= 7319) \
            or (7320 <= sic <= 7329) \
            or (7330 <= sic <= 7339) \
            or (7340 <= sic <= 7342) \
            or (7349 <= sic <= 7349) \
            or (7350 <= sic <= 7351) \
            or (7352 <= sic <= 7352) \
            or (7353 <= sic <= 7353) \
            or (7359 <= sic <= 7359) \
            or (7360 <= sic <= 7369) \
            or (7370 <= sic <= 7372) \
            or (7374 <= sic <= 7374) \
            or (7375 <= sic <= 7375) \
            or (7376 <= sic <= 7376) \
            or (7377 <= sic <= 7377) \
            or (7378 <= sic <= 7378) \
            or (7379 <= sic <= 7379) \
            or (7380 <= sic <= 7380) \
            or (7381 <= sic <= 7382) \
            or (7383 <= sic <= 7383) \
            or (7384 <= sic <= 7384) \
            or (7385 <= sic <= 7385) \
            or (7389 <= sic <= 7390) \
            or (7391 <= sic <= 7391) \
            or (7392 <= sic <= 7392) \
            or (7393 <= sic <= 7393) \
            or (7394 <= sic <= 7394) \
            or (7395 <= sic <= 7395) \
            or (7396 <= sic <= 7396) \
            or (7397 <= sic <= 7397) \
            or (7399 <= sic <= 7399) \
            or (7500 <= sic <= 7500) \
            or (7510 <= sic <= 7519) \
            or (7520 <= sic <= 7529) \
            or (7530 <= sic <= 7539) \
            or (7540 <= sic <= 7549) \
            or (7600 <= sic <= 7600) \
            or (7620 <= sic <= 7620) \
            or (7622 <= sic <= 7622) \
            or (7623 <= sic <= 7623) \
            or (7629 <= sic <= 7629) \
            or (7630 <= sic <= 7631) \
            or (7640 <= sic <= 7641) \
            or (7690 <= sic <= 7699) \
            or (8100 <= sic <= 8199) \
            or (8200 <= sic <= 8299) \
            or (8300 <= sic <= 8399) \
            or (8400 <= sic <= 8499) \
            or (8600 <= sic <= 8699) \
            or (8700 <= sic <= 8700) \
            or (8710 <= sic <= 8713) \
            or (8720 <= sic <= 8721) \
            or (8730 <= sic <= 8734) \
            or (8740 <= sic <= 8748) \
            or (8800 <= sic <= 8899) \
            or (8900 <= sic <= 8910) \
            or (8911 <= sic <= 8911) \
            or (8920 <= sic <= 8999):
        ffi30 = 22
        desc = 'Personal and Business Services'
    elif (3570 <= sic <= 3579) \
            or (3622 <= sic <= 3622) \
            or (3661 <= sic <= 3661) \
            or (3662 <= sic <= 3662) \
            or (3663 <= sic <= 3663) \
            or (3664 <= sic <= 3664) \
            or (3665 <= sic <= 3665) \
            or (3666 <= sic <= 3666) \
            or (3669 <= sic <= 3669) \
            or (3670 <= sic <= 3679) \
            or (3680 <= sic <= 3680) \
            or (3681 <= sic <= 3681) \
            or (3682 <= sic <= 3682) \
            or (3683 <= sic <= 3683) \
            or (3684 <= sic <= 3684) \
            or (3685 <= sic <= 3685) \
            or (3686 <= sic <= 3686) \
            or (3687 <= sic <= 3687) \
            or (3688 <= sic <= 3688) \
            or (3689 <= sic <= 3689) \
            or (3695 <= sic <= 3695) \
            or (3810 <= sic <= 3810) \
            or (3811 <= sic <= 3811) \
            or (3812 <= sic <= 3812) \
            or (3820 <= sic <= 3820) \
            or (3821 <= sic <= 3821) \
            or (3822 <= sic <= 3822) \
            or (3823 <= sic <= 3823) \
            or (3824 <= sic <= 3824) \
            or (3825 <= sic <= 3825) \
            or (3826 <= sic <= 3826) \
            or (3827 <= sic <= 3827) \
            or (3829 <= sic <= 3829) \
            or (3830 <= sic <= 3839) \
            or (7373 <= sic <= 7373):
        ffi30 = 23
        desc = 'Business Equipment'
    elif (2440 <= sic <= 2449) \
            or (2520 <= sic <= 2549) \
            or (2600 <= sic <= 2639) \
            or (2640 <= sic <= 2659) \
            or (2670 <= sic <= 2699) \
            or (2760 <= sic <= 2761) \
            or (3220 <= sic <= 3221) \
            or (3410 <= sic <= 3412) \
            or (3950 <= sic <= 3955):
        ffi30 = 24
        desc = 'Business Supplies and Shipping Containers'
    elif (4000 <= sic <= 4013) \
            or (4040 <= sic <= 4049) \
            or (4100 <= sic <= 4100) \
            or (4110 <= sic <= 4119) \
            or (4120 <= sic <= 4121) \
            or (4130 <= sic <= 4131) \
            or (4140 <= sic <= 4142) \
            or (4150 <= sic <= 4151) \
            or (4170 <= sic <= 4173) \
            or (4190 <= sic <= 4199) \
            or (4200 <= sic <= 4200) \
            or (4210 <= sic <= 4219) \
            or (4220 <= sic <= 4229) \
            or (4230 <= sic <= 4231) \
            or (4240 <= sic <= 4249) \
            or (4400 <= sic <= 4499) \
            or (4500 <= sic <= 4599) \
            or (4600 <= sic <= 4699) \
            or (4700 <= sic <= 4700) \
            or (4710 <= sic <= 4712) \
            or (4720 <= sic <= 4729) \
            or (4730 <= sic <= 4739) \
            or (4740 <= sic <= 4749) \
            or (4780 <= sic <= 4780) \
            or (4782 <= sic <= 4782) \
            or (4783 <= sic <= 4783) \
            or (4784 <= sic <= 4784) \
            or (4785 <= sic <= 4785) \
            or (4789 <= sic <= 4789):
        ffi30 = 25
        desc = 'Transportation'
    elif (5000 <= sic <= 5000) \
            or (5010 <= sic <= 5015) \
            or (5020 <= sic <= 5023) \
            or (5030 <= sic <= 5039) \
            or (5040 <= sic <= 5042) \
            or (5043 <= sic <= 5043) \
            or (5044 <= sic <= 5044) \
            or (5045 <= sic <= 5045) \
            or (5046 <= sic <= 5046) \
            or (5047 <= sic <= 5047) \
            or (5048 <= sic <= 5048) \
            or (5049 <= sic <= 5049) \
            or (5050 <= sic <= 5059) \
            or (5060 <= sic <= 5060) \
            or (5063 <= sic <= 5063) \
            or (5064 <= sic <= 5064) \
            or (5065 <= sic <= 5065) \
            or (5070 <= sic <= 5078) \
            or (5080 <= sic <= 5080) \
            or (5081 <= sic <= 5081) \
            or (5082 <= sic <= 5082) \
            or (5083 <= sic <= 5083) \
            or (5084 <= sic <= 5084) \
            or (5085 <= sic <= 5085) \
            or (5086 <= sic <= 5087) \
            or (5088 <= sic <= 5088) \
            or (5090 <= sic <= 5090) \
            or (5091 <= sic <= 5092) \
            or (5093 <= sic <= 5093) \
            or (5094 <= sic <= 5094) \
            or (5099 <= sic <= 5099) \
            or (5100 <= sic <= 5100) \
            or (5110 <= sic <= 5113) \
            or (5120 <= sic <= 5122) \
            or (5130 <= sic <= 5139) \
            or (5140 <= sic <= 5149) \
            or (5150 <= sic <= 5159) \
            or (5160 <= sic <= 5169) \
            or (5170 <= sic <= 5172) \
            or (5180 <= sic <= 5182) \
            or (5190 <= sic <= 5199):
        ffi30 = 26
        desc = 'Wholesale'
    elif (5200 <= sic <= 5200) \
            or (5210 <= sic <= 5219) \
            or (5220 <= sic <= 5229) \
            or (5230 <= sic <= 5231) \
            or (5250 <= sic <= 5251) \
            or (5260 <= sic <= 5261) \
            or (5270 <= sic <= 5271) \
            or (5300 <= sic <= 5300) \
            or (5310 <= sic <= 5311) \
            or (5320 <= sic <= 5320) \
            or (5330 <= sic <= 5331) \
            or (5334 <= sic <= 5334) \
            or (5340 <= sic <= 5349) \
            or (5390 <= sic <= 5399) \
            or (5400 <= sic <= 5400) \
            or (5410 <= sic <= 5411) \
            or (5412 <= sic <= 5412) \
            or (5420 <= sic <= 5429) \
            or (5430 <= sic <= 5439) \
            or (5440 <= sic <= 5449) \
            or (5450 <= sic <= 5459) \
            or (5460 <= sic <= 5469) \
            or (5490 <= sic <= 5499) \
            or (5500 <= sic <= 5500) \
            or (5510 <= sic <= 5529) \
            or (5530 <= sic <= 5539) \
            or (5540 <= sic <= 5549) \
            or (5550 <= sic <= 5559) \
            or (5560 <= sic <= 5569) \
            or (5570 <= sic <= 5579) \
            or (5590 <= sic <= 5599) \
            or (5600 <= sic <= 5699) \
            or (5700 <= sic <= 5700) \
            or (5710 <= sic <= 5719) \
            or (5720 <= sic <= 5722) \
            or (5730 <= sic <= 5733) \
            or (5734 <= sic <= 5734) \
            or (5735 <= sic <= 5735) \
            or (5736 <= sic <= 5736) \
            or (5750 <= sic <= 5799) \
            or (5900 <= sic <= 5900) \
            or (5910 <= sic <= 5912) \
            or (5920 <= sic <= 5929) \
            or (5930 <= sic <= 5932) \
            or (5940 <= sic <= 5940) \
            or (5941 <= sic <= 5941) \
            or (5942 <= sic <= 5942) \
            or (5943 <= sic <= 5943) \
            or (5944 <= sic <= 5944) \
            or (5945 <= sic <= 5945) \
            or (5946 <= sic <= 5946) \
            or (5947 <= sic <= 5947) \
            or (5948 <= sic <= 5948) \
            or (5949 <= sic <= 5949) \
            or (5950 <= sic <= 5959) \
            or (5960 <= sic <= 5969) \
            or (5970 <= sic <= 5979) \
            or (5980 <= sic <= 5989) \
            or (5990 <= sic <= 5990) \
            or (5992 <= sic <= 5992) \
            or (5993 <= sic <= 5993) \
            or (5994 <= sic <= 5994) \
            or (5995 <= sic <= 5995) \
            or (5999 <= sic <= 5999):
        ffi30 = 27
        desc = 'Retail'
    elif (5800 <= sic <= 5819) \
            or (5820 <= sic <= 5829) \
            or (5890 <= sic <= 5899) \
            or (7000 <= sic <= 7000) \
            or (7010 <= sic <= 7019) \
            or (7040 <= sic <= 7049) \
            or (7213 <= sic <= 7213):
        ffi30 = 28
        desc = 'Restaurants, Hotels, Motels'
    elif (6000 <= sic <= 6000) \
            or (6010 <= sic <= 6019) \
            or (6020 <= sic <= 6020) \
            or (6021 <= sic <= 6021) \
            or (6022 <= sic <= 6022) \
            or (6023 <= sic <= 6024) \
            or (6025 <= sic <= 6025) \
            or (6026 <= sic <= 6026) \
            or (6027 <= sic <= 6027) \
            or (6028 <= sic <= 6029) \
            or (6030 <= sic <= 6036) \
            or (6040 <= sic <= 6059) \
            or (6060 <= sic <= 6062) \
            or (6080 <= sic <= 6082) \
            or (6090 <= sic <= 6099) \
            or (6100 <= sic <= 6100) \
            or (6110 <= sic <= 6111) \
            or (6112 <= sic <= 6113) \
            or (6120 <= sic <= 6129) \
            or (6130 <= sic <= 6139) \
            or (6140 <= sic <= 6149) \
            or (6150 <= sic <= 6159) \
            or (6160 <= sic <= 6169) \
            or (6170 <= sic <= 6179) \
            or (6190 <= sic <= 6199) \
            or (6200 <= sic <= 6299) \
            or (6300 <= sic <= 6300) \
            or (6310 <= sic <= 6319) \
            or (6320 <= sic <= 6329) \
            or (6330 <= sic <= 6331) \
            or (6350 <= sic <= 6351) \
            or (6360 <= sic <= 6361) \
            or (6370 <= sic <= 6379) \
            or (6390 <= sic <= 6399) \
            or (6400 <= sic <= 6411) \
            or (6500 <= sic <= 6500) \
            or (6510 <= sic <= 6510) \
            or (6512 <= sic <= 6512) \
            or (6513 <= sic <= 6513) \
            or (6514 <= sic <= 6514) \
            or (6515 <= sic <= 6515) \
            or (6517 <= sic <= 6519) \
            or (6520 <= sic <= 6529) \
            or (6530 <= sic <= 6531) \
            or (6532 <= sic <= 6532) \
            or (6540 <= sic <= 6541) \
            or (6550 <= sic <= 6553) \
            or (6590 <= sic <= 6599) \
            or (6610 <= sic <= 6611) \
            or (6700 <= sic <= 6700) \
            or (6710 <= sic <= 6719) \
            or (6720 <= sic <= 6722) \
            or (6723 <= sic <= 6723) \
            or (6724 <= sic <= 6724) \
            or (6725 <= sic <= 6725) \
            or (6726 <= sic <= 6726) \
            or (6730 <= sic <= 6733) \
            or (6740 <= sic <= 6779) \
            or (6790 <= sic <= 6791) \
            or (6792 <= sic <= 6792) \
            or (6793 <= sic <= 6793) \
            or (6794 <= sic <= 6794) \
            or (6795 <= sic <= 6795) \
            or (6798 <= sic <= 6798) \
            or (6799 <= sic <= 6799):
        ffi30 = 29
        desc = 'Banking, Insurance, Real Estate, Trading'
    elif (4950 <= sic <= 4959) \
            or (4960 <= sic <= 4961) \
            or (4970 <= sic <= 4971) \
            or (4990 <= sic <= 4991):
        ffi30 = 30
        desc = 'Everything Else'
    else:
        ffi30 = pd.NA
        desc = ''
    if getdesc:
        return desc
    else:
        return ffi30


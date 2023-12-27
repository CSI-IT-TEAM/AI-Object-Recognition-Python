from datetime import datetime, timedelta

# Hàm xử lý log file vào thư mục LOG
_date = datetime.now() + timedelta(days=0, hours=0)
data_log_error = '/Users/it/Desktop/Python/Main_Color_Socket/LOG/data_log_error_'+_date.strftime('%Y_%m_%d')+ '.txt'
data_log = '/Users/it/Desktop/Python/Main_Color_Socket/LOG/data_log_'+_date.strftime('%Y_%m_%d')+ '.txt'

def sbLogWrite(tpye,strlog):
    return 0
    global data_log_error #tên của file log
    global data_log #tên của file log
    #lấy ngày tháng hiện tại khi log file
    _date = datetime.now() + timedelta(days=0, hours=0)
    string_date = _date.strftime('%Y-%m-%d %H:%M:%S')
    try:
        # thêm thời gian vào chuỗi string
        strlog = strlog.encode('utf-8', 'replace')
        _str = string_date + " --> " + str(strlog)
        if tpye == "ERROR":
            f = open(data_log_error,'a')
            f.write(_str + "\n")
            f.close()
        else:
            f = open(data_log,'a')
            f.write(_str + "\n")
            f.close()
        print('sbLogWrite:'+_str) 
    except Exception as e:
        print('except:'+ str(e) ) 
        print('except:'+ str(strlog) ) 
        pass
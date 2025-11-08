start=820
end=880
with open(''Untitled-1.py'',''r'',encoding=''utf-8'') as f:
    for idx,line in enumerate(f,1):
        if start<=idx<=end:
            print(f"{idx}:{line.rstrip()}" )
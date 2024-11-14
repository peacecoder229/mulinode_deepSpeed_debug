#!/usr/bin/env python3
import argparse
import pandas as pd

def process_line(line, printraw):
    #print(line)
    parts = [part for part in line.strip().split(',') if part]
    parts = parts[:-1]
    #print(parts)
    common_elements = parts[:8]

    first_instance = parts[8] if len(parts) > 8 else {}
    last_instance = parts[-1]
    #print(first_instance)
    token_value = extract_value(first_instance, 'token') if 'token' in first_instance else -1
    bs_value = extract_value(first_instance, 'BS') if 'BS' in first_instance else -1
    gflops = extract_value(first_instance, 'gflops') if 'gflops' in first_instance else -1
    sigmem = extract_value(first_instance, 'sigmem') if 'sigmem' in first_instance else -1
    totmem = extract_value(first_instance, 'totmem') if 'totmem' in first_instance else -1
    gflops = extract_value(last_instance, 'gflops') if 'gflops' in last_instance else -1
    firstlat = extract_value(last_instance, 'firstlat') if 'firstlat' in last_instance else -1
    seclat = extract_value(last_instance, 'seclat') if 'seclat' in last_instance else -1
    amx = extract_value(last_instance, 'AMX') if 'AMX' in last_instance else -1
    cache =  extract_value(last_instance, 'cache') if 'cache' in last_instance else -1
# below only parsing the parts which has inftime as key.. So the tag is not picked to do .. so the line part != "new" does not matter.
    inftime_values = [extract_value(part, 'inftime') for part in parts[8:] if 'inftime' in part]
    avg_inftime = sum(float(val) for val in inftime_values) / len(inftime_values) if inftime_values else -1
    print(parts)
    tok_lat = (avg_inftime / (int(token_value) * int(bs_value))) * 1000
    #QPS = (int(token_value) * int(parts[5]) * int(bs_value)) / avg_inftime
    QPS = (int(parts[5]) * int(bs_value)) / avg_inftime

    result = common_elements + [token_value] + [str(avg_inftime)] + [str(tok_lat), str(QPS), str(sigmem), str(totmem), str(gflops), str(firstlat), str(seclat), amx, cache]
    
    if printraw:
        result.extend(inftime_values)
    
    return result

def extract_value(instance, key):
    for item in instance.split():
        k, v = item.split('=')
        if k == key:
            return v
    return ''

def main(inputfile, printraw, group_by_cols, filters, outfile = None):

    #columns = ["drc", "cache", "memavg", "memmax", "model", "intrathread", "device", "instanceCnt", "BS", "Tokens", "Avg_lat", "tok_lat", "Throughput", "sigmemFP", "totmemFP"]
    #columns = ["memavg", "memmax", "model", "intrathread", "device", "instanceCnt", "BS", "ALLTokens", "tokens", "Avg_lat", "tok_lat", "Throughput", "sigmemFP", "totmemFP", "GFLOPS"]
    df_list = []

    with open(inputfile, 'r') as f:
        for line in f:
            processed = process_line(line, printraw)
            #print(processed)
            if printraw:
                print(' '.join(processed))
            else:
                df_list.append(processed)
    amx = processed[-2]
    cache = processed[-1]
    if amx and not cache:
        print(f"Returned AMX is:{amx}")
        columns = ["memavg", "memmax", "model", "intrathread", "device", "instanceCnt", "BS", "ALLTokens", "tokens", "Avg_lat", "tok_lat", "Throughput", "sigmemFP", "totmemFP", "GFLOPS", "Firstlat", "SecLat", "amx"]
    elif amx and cache:
        print(f"Returned AMX is:{amx}")
        columns = ["memavg", "memmax", "model", "intrathread", "device", "instanceCnt", "BS", "ALLTokens", "tokens", "Avg_lat", "tok_lat", "Throughput", "sigmemFP", "totmemFP", "GFLOPS", "Firstlat", "SecLat", "amx", "cache"]
    else:
        columns = ["memavg", "memmax", "model", "intrathread", "device", "instanceCnt", "BS", "ALLTokens", "tokens", "Avg_lat", "tok_lat", "Throughput", "sigmemFP", "totmemFP", "GFLOPS", "Firstlat", "SecLat"]

    if not printraw:

        df = pd.DataFrame(df_list, columns=columns)
        #df_reset = df.reset_index(drop=True)
        #print(df_reset.to_csv(sep=' '))
        #print(df)

        # Sort by specified columns

        

        if group_by_cols:
            df = df.sort_values(by=group_by_cols)

        # Filter by provided column-value pairs
        for col, val in filters.items():
            df = df[df[col] == val]

        #print(df)
        #for _, row in df.iterrows():
        #    print(' '.join(map(str, row)))
        if outfile:
            df.to_csv(outfile, index=False, sep=' ', mode='w')
        else:
            print(df.to_csv(index=False, sep=' '))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results file.")
    parser.add_argument("--inputfile", required=True, help="Path to the input results.txt file.")
    parser.add_argument("--outfile", required=False, help="Path to the outputfile")
    parser.add_argument("--printraw", action='store_true', help="Add inftime_values to the result.")
    #parser.add_argument("--group_by", nargs='+', default=[], help="Columns to sort the dataframe by.")
    parser.add_argument("--group_by", type=str, default="", help="Comma-separated columns to sort the dataframe by.")
    #parser.add_argument("--filter", nargs='*', action='append', default=[], help="Filter rows based on column=value pairs. Provide arguments in the format col1=val1 col2=val2 ...")
    parser.add_argument("--filter", type=str, default="", help="Filter rows based on column=value pairs in the format col1=val1,col2=val2,...")
    args = parser.parse_args()
    group_by_cols = args.group_by.split(",") if args.group_by else []

    #for col in group_by_cols:
    #    if col not in columns:
    #        raise ValueError(f"Column '{col}' specified in --group_by does not exist in the dataframe.")

    #filters = {item.split('=')[0]: item.split('=')[1] for sublist in args.filter for item in sublist}
    filters = {}
    if args.filter:
        filter_items = args.filter.split(',')
        for item in filter_items:
            key, value = item.split('=')
            filters[key] = value
    
    main(args.inputfile, args.printraw, group_by_cols, filters, args.outfile)


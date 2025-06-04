import os
import sys
import pygrib

def inspect_grib_file(file_path):
    """
    Print detailed information about the contents of a GRIB file.
    """
    print(f"\nInspecting GRIB file: {file_path}")
    try:
        with pygrib.open(file_path) as grbs:
            messages = [grb for grb in grbs]
            print(f"\nFound {len(messages)} messages")
            
            # Group by shortName
            by_shortname = {}
            for msg in messages:
                shortname = msg.shortName
                if shortname not in by_shortname:
                    by_shortname[shortname] = []
                by_shortname[shortname].append(msg)
            
            print("\nVariables found:")
            for shortname, msgs in by_shortname.items():
                print(f"\n{shortname}:")
                print(f"  Number of messages: {len(msgs)}")
                # Print details of first message as example
                msg = msgs[0]
                print("  Example message details:")
                print(f"    typeOfLevel: {getattr(msg, 'typeOfLevel', 'N/A')}")
                print(f"    stepType: {getattr(msg, 'stepType', 'N/A')}")
                print(f"    level: {getattr(msg, 'level', 'N/A')}")
                print(f"    dataTime: {getattr(msg, 'dataTime', 'N/A')}")
                print(f"    validityTime: {getattr(msg, 'validityTime', 'N/A')}")
                print(f"    productDefinitionTemplateNumber: {getattr(msg, 'productDefinitionTemplateNumber', 'N/A')}")
                
    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_grib.py <path_to_grib_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    inspect_grib_file(file_path) 
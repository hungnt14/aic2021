import argparse
import os
import glob
import time
from natsort import natsorted

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input1", default="")
  parser.add_argument("--input2", default="")
  parser.add_argument("--output", default="")
  return parser

if __name__ == "__main__":
  start_time = time.time()
  args = get_parser().parse_args()

  files1 = natsorted(glob.glob(args.input1 + "*"))

  for filepath1 in files1:
    filename = os.path.basename(filepath1)
    
    filepath2 = args.input2 + filename

    f1 = open(filepath1, "r", encoding="utf-8")
    lines1 = f1.read()
    f1.close()

    try:
      f2 = open(filepath2, "r", encoding="utf-8")
      lines2 = f2.read()
      f2.close()
    except:
      print("No file", filename, "- assign empty")
      lines2 = ""


    lines = lines1 + lines2

    f = open(args.output + filename, "w", encoding="utf-8")
    f.write(lines)
    f.close()
    
    print("Done merged file: ", filename)
 print("============ FINISHED FINAL MERGE (time elapsed: {}). TOTAL RECOGNIZED BBOX: {} ============".format(str(time.time() - start_time)))

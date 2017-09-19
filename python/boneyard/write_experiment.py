import argparse
import os
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--analysis', required=True)
  parser.add_argument('--server_root', required=True)
  args = parser.parse_args()

  files = os.listdir('%s/%s/json/' % (args.server_root, args.analysis))   

  json_files = sorted([f for f in files if f.endswith('json')])
  exp_subs = [tuple(f.split('_')[0:2]) for f in json_files]

  with open('%s/template.html' % args.server_root,'r') as f:
    template = f.read()


  with open('%s/%s/index.html' % (args.server_root, args.analysis),'w') as index_f:
    for i in xrange(len(json_files)):
      fname = json_files[i]
      es = exp_subs[i]
      html_fname = fname.replace('json', 'html')
      html_content = template.replace('<SUBJECT>',es[1]).replace('<EXPERIMENT>', es[0]).replace(
          "<ANALYSIS>", args.analysis);
      with open('%s/%s/%s' % (args.server_root, args.analysis, html_fname),'w') as f:
        f.write(html_content)
      index_f.write('<a href="%s">%s</a><br/>' % (html_fname, '%s_%s' % (es[0], es[1])))

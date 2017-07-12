import sys
import BaseHTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
import ssl

from camera import camare_authcode_capture
class AuthCodeRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        print self.path;
        if self.path.startswith('/get_authcode'):
            self.send_response(200)
            # code = '123456'
            while 1:
                code = camare_authcode_capture(False)
                if len(code)==6:
                    break
            content="document.getElementsByName('authCode')[0].value= '{}';\
    document.getElementById('confirm').click();\
    console.log('authcode:'+{});".format(code, code)
            self.send_header("Content-type", "text/html; charset=UTF-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            print "[DEBUG] code : {}".format(code)
            self.wfile.write(content)


HandlerClass = AuthCodeRequestHandler
ServerClass = BaseHTTPServer.HTTPServer
Protocol = "HTTP/1.0"

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8448
server_address = ('127.0.0.1', port)

HandlerClass.protocol_version = Protocol
httpd = ServerClass(server_address, HandlerClass)

sa = httpd.socket.getsockname()
print "Serving HTTP on", sa[0], "port", sa[1], "..."
# httpd.socket = ssl.wrap_socket (httpd.socket, certfile='D:/certificate.crt', keyfile='D:/privatekey.key', server_side=True)
httpd.socket = ssl.wrap_socket (httpd.socket, certfile='D:/MyCompanyLocalhost.cer', keyfile='D:/MyCompanyLocalhost.pvk', server_side=True)

# httpd.socket = ssl.wrap_socket (httpd.socket, certfile='D:/App/Anaconda2/pkgs/requests-2.14.2-py27_0/Lib/site-packages/requests/cacert.pem', server_side=True)
httpd.serve_forever()
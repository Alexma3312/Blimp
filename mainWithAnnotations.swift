//
//  mainWithAnnotations.swift
//  
//
//  Created by Swetha Rajagopalan on 9/30/19.
//


//Foundation is used to Access essential data types, collections, and operating-system services to define the base layer of functionality for your app.
import Foundation

//Private access restricts the use of an entity to the enclosing declaration, and to extensions of that declaration that are in the same file.
//let defines a constant and var defines a variable
//let: value of constant doesn't need to be defined at compile time but the value must only be declared once
//let enables compiler optimizations

//boolean types are indicated by the variable being assigned true or false; "boolean" is not explicitly mentioned
private var canWrite = false


//URLSession creates a session. A session is an open page which groups HTTP requests
//URLSession is used to create URLSessionTask instances: fetch and return data to your app and webservices
//delegate object: handles authentication calls
//delegateQueue: queue on which all operations are performed
var session: URLSession = URLSession(configuration: .default, delegate: self, delegateQueue: .main)

//takes your string URL and converts it to a URL object
let url = URL(string: "http://127.0.0.1:12345")! //replace this string with the address of the server or module that we're uploading the data to

//creating URL request
//url: URL for the request
//cache policy for the request - can requests be satisfied using cached versions of it
//timeout interval for the request: default is 60 seconds - how long the machine can remain idle
var request = URLRequest(url: url,
                         cachePolicy: .reloadIgnoringLocalCacheData,
                         timeoutInterval: 10)

//HTTP request method: what action should be performed for which resources
//GET: representation of the specified resource. Requests using GET should only retrieve data
//POST: submit an entity to the specified resource, often causing a change in state or side effects on the server
request.httpMethod = "POST"

//URLSessionUploadTask upload data in the background
let uploadTask = session.uploadTask(withStreamedRequest: request)

//resumes task if suspended
uploadTask.resume()

//bound pair of input and output streams
struct Streams {
    let input: InputStream
    let output: OutputStream
}

//guard: condition must be true for the code after the guard statement to be executed
//
var boundStreams: Streams = {
    var inputOrNil: InputStream? = nil
    var outputOrNil: OutputStream? = nil
    Stream.getBoundStreams(withBufferSize: 4096,
                           inputStream: &inputOrNil,
                           outputStream: &outputOrNil)
    guard let input = inputOrNil, let output = outputOrNil else {
        fatalError("On return of `getBoundStreams`, both `inputStream` and `outputStream` will contain non-nil streams.")
    }
    
    // configure and open output stream
    output.delegate = self
    //
    output.schedule(in: .current, forMode: .default)
    output.open()
    return Streams(input: input, output: output)
}()

func urlSession(_ session: URLSession, task: URLSessionTask,
                needNewBodyStream completionHandler: @escaping (InputStream?) -> Void) {
    completionHandler(boundStreams.input)
}

func stream(_ aStream: Stream, handle eventCode: Stream.Event) {
    guard aStream == boundStreams.output else {
        return
    }
    if eventCode.contains(.hasSpaceAvailable) {
        canWrite = true
    }
    if eventCode.contains(.errorOccurred) {
        // Close the streams and alert the user that the upload failed.
    }
}

var timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) {
    [weak self] timer in
    guard let self = self else { return }
    
    if self.canWrite {
        let message = "*** \(Date())\r\n"
        guard let messageData = message.data(using: .utf8) else { return }
        let messageCount = messageData.count
        let bytesWritten: Int = messageData.withUnsafeBytes() { (buffer: UnsafePointer<UInt8>) in
            self.canWrite = false
            return self.boundStreams.output.write(buffer, maxLength: messageCount)
        }
        if bytesWritten < messageCount {
            // Handle writing less data than expected.
        }
    }
}

//
//  main.swift
//  PoseEstimation-CoreML
//
//  Created by Swetha Rajagopalan on 5/7/19.
//  Copyright © 2019 FrankCVSPE. All rights reserved.
//

import Foundation
private var canWrite = false

var session: URLSession = URLSession(configuration: .default, delegate: self, delegateQueue: .main)


let url = URL(string: "http://127.0.0.1:12345")! //replace this string with the address of the server or module that we're uploading the data to
var request = URLRequest(url: url,
                         cachePolicy: .reloadIgnoringLocalCacheData,
                         timeoutInterval: 10)
request.httpMethod = "POST"
let uploadTask = session.uploadTask(withStreamedRequest: request)
uploadTask.resume()

struct Streams {
    let input: InputStream
    let output: OutputStream
}

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

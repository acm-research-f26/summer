using UnityEngine;
using NativeWebSocket;
using System;

[Serializable]
public class ReceivedMessage
{
    public string message_type;
    public string[] possible_actions;
}

[Serializable]
public class SentMessage
{
    public string message_type;
    public string culprit;
    
    public SentMessage(string msgType, string theCulprit)
    {
        message_type = msgType;
        culprit = theCulprit;
    }
}

public class WebsocketScript : MonoBehaviour
{
    WebSocket websocket;
    public static Action<ReceivedMessage> messageReceived;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    async void Start()
    {
        websocket = new WebSocket("ws://localhost:6767");

        websocket.OnOpen += () => Debug.Log("Connection open!");
        websocket.OnError += (e) => Debug.Log("Error! " + e);
        websocket.OnClose += (code) => Debug.Log("Connection closed!");

        websocket.OnMessage += (bytes) =>
        {
            var jsonMessage = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log("Received: " + jsonMessage);
            ReceivedMessage parsedMsg = JsonUtility.FromJson<ReceivedMessage>(jsonMessage);
            messageReceived.Invoke(parsedMsg);
        };

        await websocket.Connect();
    }

    public async void RequestAction()
    {
        SentMessage theSentMessage = new SentMessage("get_action", "");
        string jsonMsg = JsonUtility.ToJson(theSentMessage);

        await websocket.SendText(jsonMsg);
    }

    // Update is called once per frame
    void Update()
    {
    }

    private async void OnApplicationQuit()
    {
        await websocket.Close();
    }
}

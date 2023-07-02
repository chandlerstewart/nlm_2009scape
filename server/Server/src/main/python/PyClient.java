package python;

import com.google.gson.Gson;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Socket;

public class PyClient{

    private static String hostName = "localhost";
    private static int portNumber = 5000;
    private static Message message = new Message("Connected", "");


    public static void sendStatus(){

        try (
                Socket socket = new Socket(hostName, portNumber);
                DataOutputStream out = new DataOutputStream(socket.getOutputStream());
                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        ) {
            Gson gson = new Gson();

            // Send message to server
            String jsonMessage = gson.toJson(message);
            out.writeUTF(jsonMessage);

            // Receive response from server
            String jsonResponse = in.readLine();
            Message response = gson.fromJson(jsonResponse, Message.class);

            out.close();
            socket.close();

            if (response != null) {
                System.out.println("Server response: " + response.getCommand());
                message = PyCommands.ProcessCommand(response);
            }
        } catch (IOException e) {
            resetMessage();
        }
    }

    private static void resetMessage(){
        message = new Message("Connected", "");
    }

}



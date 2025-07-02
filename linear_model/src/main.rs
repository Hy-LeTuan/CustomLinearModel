use ::std::io::{self, BufRead};
use std::fs::File;
use std::path::Path;

fn main() {
    let (header, body) = read_dataset();

    println!("Header: {:?}", header);
    println!("Body: {:?}", body);
}

fn read_dataset() -> (Vec<String>, Vec<Vec<f32>>) {
    let file_path = Path::new("dataset").join("taiwan_real_estate.csv");
    let mut header: Vec<String> = Vec::new();
    let mut body: Vec<Vec<f32>> = Vec::new();

    let file = File::open(file_path);
    let file = match file {
        Ok(f) => f,
        Err(_) => {
            println!("Error in opening file, aborting now");
            return (header, body);
        }
    };

    let reader = io::BufReader::new(file);

    for (index, line) in reader.lines().into_iter().enumerate() {
        let line_extracted = match line {
            Ok(s) => s,
            Err(_) => return (header, body),
        };

        let split_line = line_extracted.split(',');

        let mut temp_body = Vec::new();

        for (chunk_index, chunk) in split_line.enumerate() {
            if index == 0 {
                header.push(String::from(chunk));
            } else {
                let value = chunk.parse::<f32>();

                // if value is not a valid number, skip that line
                let value = match value {
                    Ok(value) => value,
                    Err(_) => {
                        println!("Skipping line {index} from float parsing error");
                        break;
                    }
                };

                temp_body.push(value);
            }
        }

        if temp_body.len() > 0 {
            body.push(temp_body);
        }
    }

    return (header, body);
}

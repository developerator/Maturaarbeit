using System;
using System.Collections;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace PatternRecognition
{
    public partial class PatternRecognition : Form
    {
        NeuralNetwork network;

        static int nRows = 7;
        static int nCols = 5;
        
        //array for the grid (redCell=True, greyCell=False)
        bool[,] redCells = new bool[nCols, nRows];

        //'patterns' holds all the patterns added to the listBox (will contain bool[,])
        ArrayList patterns = new ArrayList();
        //'differentPatterns' holds all different names/classes of the listBox (will contain string)
        ArrayList differentPatterns;

        int cellSizeX = 100;
        int cellSizeY = 100;

        public PatternRecognition()
        {
            InitializeComponent();

            cellSizeX = (int) Math.Ceiling((float) Grid.Width  / nCols) + 1;
            cellSizeY = (int) Math.Ceiling((float) Grid.Height / nRows) + 1;

            //As a heuristic the number of hiddens is set by default to be the number of cells
            txtHiddens.Text = (nCols * nRows).ToString();

            //Set filed dialogs initial directory to the apps path
            openFileDialog1.InitialDirectory = Path.GetDirectoryName(Application.ExecutablePath);
        }                       

        private void Grid_Paint(object sender, PaintEventArgs e)
        {            
            Graphics g = e.Graphics;
            Brush b = new SolidBrush(Color.Red);
            Pen p = new Pen(Color.Black, 3);

            
            for (int i = 0; i < nCols; i++)                          
                for (int j = 0; j < nRows; j++)                
                    if (redCells[i, j])
                        g.FillRectangle(b, new Rectangle(i * cellSizeX, j * cellSizeY, cellSizeX, cellSizeY));
                           

            //Draw the vertical lines
            for (int i = 0; i <= nCols; i++)                                                                           
                g.DrawLine(p,   i * cellSizeX, 0, i * cellSizeX, nRows * cellSizeY);  // Vertical   //x

            //Draw the horizontal lines
            for (int i = 0; i <= nRows; i++)            
                g.DrawLine(p,   0, i * cellSizeY, nCols * cellSizeX, i * cellSizeY);  // Horizontal //y     
                  
        }

        private void Grid_MouseDown(object sender, MouseEventArgs e)
        {
            EvaluateCell(e);
        }

        private void Grid_MouseMove(object sender, MouseEventArgs e)
        {
            EvaluateCell(e);
        }

        /// <summary>
        /// This function evaluates which cells the user clicked and updates redCells accordingly in order to
        /// ensure that in Grid_Paint() the right cells are made red
        /// </summary>
        void EvaluateCell(MouseEventArgs e)
        {
            //Evaluate only if something is clicked
            if (e.Button != MouseButtons.Left && e.Button != MouseButtons.Right)
                return;

            int x = CalcXY(e.X, e.Y)[0];
            int y = CalcXY(e.X, e.Y)[1];
            //Check if number is inside range
            if (!Enumerable.Range(0, nCols).Contains(x) ||
                !Enumerable.Range(0, nRows).Contains(y))
                return;

            //Left  MouseBtn: Make cell red  (set True)
            //Right MouseBtn: Make cell grey (set False)         
            redCells[x, y] = (e.Button == MouseButtons.Left);

            //Call Grid_Paint()
            Grid.Refresh();
        }

        int[] CalcXY(params int[] p)
        {            
            int x = (int)Math.Floor((float) p[0] / cellSizeX);
            int y = (int)Math.Floor((float) p[1] / cellSizeY);

            return new int[] {x,y};
        }

        private void btnAdd_Click(object sender, EventArgs e)
        {            
            patterns.Add(redCells.Clone()); //If not clone the array is referenced directly and every change results in a change in 'patterns'!
            listBox.Items.Add(txtAdd.Text);
        }                

        private void btnLearn_Click(object sender, EventArgs e)
        {            
            //Find the number of different patterns
            differentPatterns = new ArrayList();
            for (int i = 0; i < listBox.Items.Count; i++)
            {
                if (!differentPatterns.Contains(listBox.Items[i].ToString()))
                    differentPatterns.Add(listBox.Items[i].ToString());
            }

            //Init the network
            network = new NeuralNetwork(nCols * nRows, int.Parse(txtHiddens.Text), differentPatterns.Count, txtLearn);

            //outside big loop to save time
            double[][] _Patterns = new double[patterns.Count][];
            double[][] _targetOuts = new double[patterns.Count][];
            for (int i = 0; i < patterns.Count; i++)
            {
                _Patterns[i] = boolToDouble((bool[,])patterns[i]);
                _targetOuts[i] = new double[differentPatterns.Count];
                _targetOuts[i][differentPatterns.IndexOf(listBox.Items[i].ToString())] = 1;
            }

            //Big "Learning-Loop":
            for (int i = 0; i < int.Parse(txtRounds.Text); i++)
            {
                for (int j = 0; j < patterns.Count; j++)
                {
                    double[] pattern = _Patterns[j];
              
                    double[] targetOut = _targetOuts[j];

                    network.Train(pattern);
                    network.Backpropagate(targetOut);
                }               
            }
        }

        private void btnRecognise_Click(object sender, EventArgs e)
        {
            if (network == null)
            {
                MessageBox.Show("Firstly learning is needed!");
                return;
            }

            double[] result = network.Compute(boolToDouble(redCells));
            string answer = differentPatterns[Array.IndexOf(result, result.Max())].ToString();

            if (1 - result.Max() <= float.Parse(txtErr.Text))
                MessageBox.Show(answer);
            else
                MessageBox.Show("Sorry, I'm not very sure. \n Probably: " + answer);
        }

        private void btnRemove_Click(object sender, EventArgs e)
        {
            if (listBox.SelectedItem == null)
            {
                MessageBox.Show("Please select the item you want to remove");
                return;
            }

            int selected = listBox.SelectedIndex;

            patterns.RemoveAt(selected);
            listBox.Items.RemoveAt(selected);
        }

        private void listBox_DoubleClick(object sender, EventArgs e)
        {
            if (listBox.SelectedItem != null)
            {
                redCells = (bool[,])((bool[,])patterns[listBox.SelectedIndex]).Clone();//Clone wieder einmal wichtig!
                Grid.Refresh();
            }                
        }

        private void btn_load_Click(object sender, EventArgs e)
        {            
            if (openFileDialog1.ShowDialog() != DialogResult.OK)
                return;
                        

            if (File.Exists(openFileDialog1.FileName))
            {
                string[] content = File.ReadAllLines(openFileDialog1.FileName);

                //Reset everything
                listBox.Items.Clear();
                redCells = new bool[nCols, nRows];
                patterns = new ArrayList();
                differentPatterns = null;
                network = null;        


                for (int i = 0; i < content.Length; i++)
                {
                    if ((i+1) % 2 != 0)
                    {
                        //Labels
                        listBox.Items.Add(content[i]);
                    }
                    else
                    {
                        //Content
                        string[] numbers = content[i].Split(',');
                        double[] doublepattern = new double[nRows * nCols];

                        for (int j = 0; j < doublepattern.Length; j++)                        
                            doublepattern[j] = double.Parse(numbers[j]);

                        patterns.Add(doubleToBool(doublepattern));                        
                    }
                }

                Grid.Refresh();
            }
        }

        private void btn_save_Click(object sender, EventArgs e)
        {
            if (!File.Exists("Save.txt"))
                SaveFile();
            else
            {
                var confirmed = MessageBox.Show("Do you want to overwrite the existing file?", "Confirm", MessageBoxButtons.YesNo);

                if (confirmed == DialogResult.Yes)
                    SaveFile();
            }                           
        }

        void SaveFile()
        {
            // Create a file to write to.
            using (StreamWriter sw = File.CreateText("Save.txt"))
            {
                for (int i = 0; i < listBox.Items.Count; i++)
                {
                    //Label
                    sw.WriteLine(listBox.Items[i].ToString());
                    double[] val = boolToDouble((bool[,])patterns[i]);

                    //Content
                    StringBuilder sb = new StringBuilder();                    
                    foreach (object o in val)
                    {
                        string num = Convert.ToString(o);
                        sb.Append(num+",");                        
                    }
                    sw.WriteLine(sb.ToString());
                }                
            }
            
        }

        /// <summary>
        /// This function converts a 2-dimensional bool array with nCols cols and nRows rows (format to be displayed in the grid)
        /// into a 1-dimensional double array containg only the values 1 and zero (input for network)
        /// </summary>
        double[] boolToDouble(bool[,] b)
        {
            double[] result = new double[nRows * nCols];


            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    if (b[j, i] == true)
                        result[i * nCols + j] = 1;
                    else
                        result[i * nCols + j] = -1;
                }
            }

            return result;
        }

        /// <summary>
        /// This function converts a 1-dimensional double array containg only the values 1 and zero (input for network)
        /// into a 2-dimensional bool array with nCols cols and nRows rows (format to be displayed in the grid)
        /// </summary>
        bool[,] doubleToBool(double[] d)
        {
            bool[,] result = new bool[nCols, nRows];

            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    if (d[i * nCols + j] == 1)
                        result[j, i] = true;
                    else
                        result[j, i] = false;
                }
            }

            return result;
        }
    }
}
